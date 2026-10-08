"""
Interoperability between :class:`~kingdon.multivector.MultiVector` and :code:`torch`: a torch function is handed :code:`mv.values()`,
whose leading blade axis every :code:`torch.nn` module treats as a batch axis, unless the multivector has an operation of that name,
which torch's name then means. :code:`Algebra(..., backend='torch')` imports this module, which puts :code:`MultiVector.__torch_function__`
in place. See :doc:`backends/torch`.
"""
from __future__ import annotations

import inspect
from functools import cache

import sympy
import sympy.printing.pytorch
import torch

from kingdon.codegen import FloatConstants
from kingdon.multivector import MultiVector

try:
    # Whoever works with torch wants einops.reduce and friends too, and should not have to find a
    # second import for them. Nothing here needs einops, so its absence is not an error.
    import kingdon.einops_backend  # noqa: F401
except ImportError:  # pragma: no cover
    pass


def cat_blades(arrays):
    """ Arrays whose first axis is the blade axis joined along it, with their other axes broadcast against each other: a bias of shape (o,) joins blades of (batch, o), and a list of numbers a blade each. """
    like = next(a for a in arrays if isinstance(a, torch.Tensor))
    arrays = [a if isinstance(a, torch.Tensor) else _constant(tuple(a), like.device, like.dtype) for a in arrays]
    shape = torch.broadcast_shapes(*(a.shape[1:] for a in arrays))
    return torch.cat([a.reshape(len(a), *(1,) * (len(shape) + 1 - a.ndim), *a.shape[1:]).expand(len(a), *shape) for a in arrays])


def take(array, index: tuple):
    """ The entries along the first axis of `array` that the nested tuple `index` holds. """
    return array[_constant(index, array.device)]


@cache
def _constant(values: tuple, device, dtype=None) -> torch.Tensor:
    """ `values` as a tensor on `device`, made once: a list would be copied to the device, and waited for, at every call. """
    return torch.tensor(values, dtype=dtype, device=device)


class TorchPrinter(FloatConstants, sympy.printing.pytorch.TorchPrinter):
    """
    Prints an operator whose codegen_symbolcls is a sympy symbol, and which can therefore call sympy's functions -- :code:`erf`, say -- as torch code,
    and the einops nodes of :mod:`kingdon.codegen` in it. values_asarray is in the namespace of every function generated for an algebra over torch.
    """
    namespace = {'torch': torch, 'cat_blades': cat_blades, 'take': take}

    def _print_Stack(self, e):
        if len(e.args) == 1 and not e.args[0].is_number:
            return f"({self._print(e.args[0])})[None]"  # A view.
        return f"values_asarray([{', '.join(map(self._print, e.args))}])"

    def _print_Cat(self, e):
        return f"cat_blades([{', '.join(map(self._print, e.args))}])"

    def _print_Blades(self, e):
        array, start, stop, *_ = e.args
        return f"{self._print(array)}[{start}:{stop}]"

    def _print_BladeSum(self, e):
        return f"torch.sum({self._print(e.args[0])}, 0)"

    def _print_Split(self, e):
        array, sizes = e.args
        return f"torch.split({self._print(array)}, {list(map(int, sizes))})"

    def _print_Unbind(self, e):
        return f"torch.unbind({self._print(e.args[0])})"

    def _print_Item(self, e):
        pieces, i = e.args
        return f"{self._print(pieces)}[{i}]"

    def _print_Einsum(self, e):
        pattern, *operands = e.args
        return f"torch.einsum({pattern.name!r}, {', '.join(map(self._print, operands))})"

    def _print_Reduce(self, e):
        array, operation, axes = e.args
        name = {'max': 'amax', 'min': 'amin'}.get(operation.name, operation.name)
        return f"torch.{name}({self._print(array)}, dim={tuple(map(int, axes))})"

    def _print_Reshape(self, e):
        array, k, sizes = e.args
        return f"torch.unflatten(torch.flatten({self._print(array)}, {-int(k)}), -1, {tuple(map(int, sizes))})"

    def _print_Take(self, e):
        array, index = e.args
        return f"take({self._print(array)}, {_nested(index)})"


def _nested(index):
    """ A sympy Tuple of integers, or of such Tuples, as python's. """
    return tuple(map(_nested, index)) if isinstance(index, sympy.Tuple) else int(index)


def values_asarray(values):
    """
    The coefficients of a multivector as a single tensor whose first axis is the blade axis: the :code:`values_asarray` of
    :code:`Algebra(..., backend='torch')`, which you may also pass on its own::

        >>> alg = Algebra.fromname('3DPGA', values_asarray=values_asarray)

    Coefficients that do not agree are broadcast against each other, so that a number a type's layout contributes, the :code:`1.0`
    of a normalized :code:`Point` say, does not stop a multivector from having a shape. Which case it is, is decided by looking at
    the values rather than by catching what :code:`torch.stack` raises, since an exception out of a torch call breaks the graph of
    :code:`torch.compile`. Values without any tensor, as the plain :code:`1` every basis blade is built from, are left as they are.
    """
    if not isinstance(values, (list, tuple)):
        return values
    like = next((v for v in values if isinstance(v, torch.Tensor)), None)
    if like is None:
        return values
    if all(isinstance(v, torch.Tensor) and v.shape == like.shape and v.device == like.device
           for v in values):
        return torch.stack(values)  # torch promotes the dtypes itself.
    return torch.stack(torch.broadcast_tensors(
        *(torch.as_tensor(v, dtype=like.dtype, device=like.device) for v in values)))


#: What a torch name means for a multivector, as :code:`{torch name: the name kingdon has for it}`.
#: The operators are the ones the two spell differently, and the last four are torch's dunders,
#: since it has no name of its own for :code:`|` and friends. The rest are asked of
#: :class:`~kingdon.multivector.MultiVector`, so that the rule follows kingdon rather than a list
#: here going stale: every operation it has that torch has a name for.
_OPERATIONS = {'add': 'add', 'sub': 'sub', 'subtract': 'sub', 'mul': 'gp', 'multiply': 'gp',
               'div': 'div', 'divide': 'div', 'true_divide': 'div', 'neg': 'neg',
               'negative': 'neg', 'matmul': 'proj', '__or__': 'ip', '__xor__': 'op',
               '__and__': 'rp', '__rshift__': 'sw'}
_OPERATIONS.update({name: name for name in dir(MultiVector)
                    if name not in _OPERATIONS and not name.startswith('_')
                    and getattr(torch, name, None) is not None})


def _operation(name, kingdon, func):
    """
    Hand torch's `name` to kingdon under the name `kingdon` has for it.

    The algebra performs it where it has it, since only the algebra takes the operands in the order
    torch had them: :code:`tensor | mv` is the inner product as much as :code:`mv | tensor` is.
    Whatever else `func` offers, the kingdon operation has no room for, and says so. Note that torch
    forwards the defaults it was not given, :code:`torch.norm` among them, so those are compared
    rather than counted.
    """
    try:
        defaults = {p.name: p.default for p in inspect.signature(func).parameters.values()
                    if p.default is not inspect.Parameter.empty}
    except (TypeError, ValueError):  # A builtin without a signature, which forwards nothing.
        defaults = {}

    def handler(*operands, **kwargs):
        if given := [key for key, value in kwargs.items() if value is not defaults.get(key)]:
            raise TypeError(f'{name} of a MultiVector is its {kingdon}, which takes no '
                            f'{given[0]}. Use mv.values() if you mean the coefficients.')
        mv = next(operand for operand in operands if isinstance(operand, MultiVector))
        return getattr(mv.algebra, kingdon, getattr(MultiVector, kingdon))(*operands)
    return handler


#: The functions with a handler of their own, as :code:`{torch function: handler}`.
_HANDLED = {func: _operation(name, kingdon, func)
            for name, kingdon in _OPERATIONS.items() for namespace in (torch, torch.Tensor)
            if (func := getattr(namespace, name, None)) is not None}


def torch_function(func, types, args=(), kwargs=None):
    """
    Implementation of :code:`MultiVector.__torch_function__`.
    """
    kwargs = kwargs or {}
    name = getattr(func, '__name__', func)
    if not all(issubclass(t, (MultiVector, torch.Tensor)) for t in types):
        return NotImplemented  # Some other type may know what to do with this.
    if (handler := _HANDLED.get(func)) is not None:
        return handler(*args, **kwargs)

    mvs = [arg for arg in (*args, *kwargs.values()) if isinstance(arg, MultiVector)]
    if not mvs:
        raise TypeError(
            f'{name} has no multivector of its own among its arguments, only ones tucked away in '
            'a sequence, and kingdon cannot tell how their blades should line up. einops speaks '
            'multivector: use einops.pack or einops.einsum instead.')

    unwrap = lambda arg: values_asarray(arg.values()) if isinstance(arg, MultiVector) else arg
    values = func(*map(unwrap, args), **{key: unwrap(v) for key, v in kwargs.items()})
    if not isinstance(values, torch.Tensor):
        return values  # torch.allclose and the like, which do not give back coefficients.

    mv, blades = mvs[0], len(mvs[0].keys())
    if values.shape[:1] != (blades,):
        raise TypeError(
            f'{name} left {tuple(values.shape[:1])} where the {blades} blades of this '
            f'{type(mv).__name__} were, so it addressed the blade axis: torch counts the axes of '
            'mv.values(), which are the blade axis and then those of mv.shape. To address the '
            'axes of the multivector itself use einops.reduce or einops.rearrange, whose patterns '
            'refer to mv.shape.')
    return type(mv).fromkeysvalues(mv.algebra, mv.keys(), values, raw=True)


def torch_getattr(mv: MultiVector, name: str):
    """
    Implementation of the torch half of :code:`MultiVector.__getattr__`, called for an attribute that is not a basis
    blade once torch has been imported.

    A tensor method never reaches :code:`MultiVector.__torch_function__`,
    because the descriptor turns down a non-tensor before dispatch gets a chance. So :code:`mv.to`,
    :code:`mv.detach` and :code:`mv.relu` are resolved here instead, and go through
    :func:`torch_function` like their free function counterparts do. An attribute that is not a
    method -- :code:`mv.dtype`, :code:`mv.device`, :code:`mv.grad` -- describes the coefficients.

    :raises AttributeError: if a tensor has no `name` either.
    """
    attribute = getattr(torch.Tensor, name, None)
    if attribute is None or not isinstance(next(iter(mv.values()), None), torch.Tensor):
        raise AttributeError(f'{type(mv).__name__} object has no attribute or basis blade {name}')
    if not callable(attribute):
        return attribute.__get__(values_asarray(mv.values()))
    # The tensor method, not its free function namesake, so that mv.foo() takes its arguments the
    # way torch.Tensor.foo does. _HANDLED knows both spellings of the operators.
    return lambda *args, **kwargs: torch_function(attribute, (type(mv),), (mv, *args), kwargs)


_pytree_registered = set()


def register_pytree_nodes(types):
    """
    Register multivector `types` with :code:`torch.utils._pytree`, so that :code:`torch.export`, which flattens whatever crosses
    its boundary, can take and give back a multivector. :code:`torch.compile` traces through one as the python object it is.

    The coefficients are the only child; the type, the algebra and the keys are static context, so the sparsity of a multivector
    is a constant the graph specializes on. Export hashes that context, hence :code:`Algebra.__hash__`. Types are registered per
    algebra, since an algebra generates classes of its own for the layouts it is given.

    :param types: multivector classes to register. Registering one twice is a no-op.
    """
    from torch.utils._pytree import SequenceKey, register_pytree_node

    def flatten(mv):
        return [mv._values], (type(mv), mv.algebra, mv._keys)

    def flatten_with_keys(mv):
        return [(SequenceKey(0), mv._values)], (type(mv), mv.algebra, mv._keys)

    def unflatten(values, context):
        cls, algebra, keys = context
        return cls.fromkeysvalues(algebra, keys, next(iter(values)))

    for cls in types:
        if cls not in _pytree_registered:
            register_pytree_node(cls, flatten, unflatten, flatten_with_keys_fn=flatten_with_keys)
            _pytree_registered.add(cls)


_kingdon_getattr = MultiVector.__getattr__


def _getattr(self, name):
    """ :code:`MultiVector.__getattr__`, with torch behind the basis blades. """
    try:
        return _kingdon_getattr(self, name)
    except AttributeError:
        # Private names are left alone, so copy and pickle cannot recurse into a half built mv.
        if name.startswith('_'):
            raise
        return torch_getattr(self, name)


def _torch_function(cls, func, types, args=(), kwargs=None):
    """ :code:`MultiVector.__torch_function__`; the work is in :func:`torch_function`. """
    return torch_function(func, types, args, kwargs or {})


# Importing this module is what opts a multivector in to torch, and an
# :class:`~kingdon.algebra.Algebra` with :code:`backend='torch'` is what imports it. Until then
# :class:`~kingdon.multivector.MultiVector` has no :code:`__torch_function__` at all, so torch does
# not treat it as a type that overrides anything and :code:`tensor * mv` falls through to
# :meth:`~kingdon.multivector.MultiVector.__rmul__` exactly as it does without torch installed.
MultiVector.__getattr__ = _getattr
MultiVector.__torch_function__ = classmethod(_torch_function)
