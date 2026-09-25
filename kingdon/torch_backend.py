"""
Interoperability between :class:`~kingdon.multivector.MultiVector` and :code:`torch`, so that a
multivector over torch coefficients can be handed to a :code:`torch.nn` module directly::

    >>> import torch
    >>> from kingdon import Algebra
    >>> alg = Algebra.fromname('3DPGA', backend='torch')
    >>> p = alg.point(torch.randn(3, 32, 4))
    >>> p.shape
    Point[(32, 4)]
    >>> torch.nn.Sequential(torch.nn.Linear(4, 10), torch.nn.GELU())(p).shape
    Point[(32, 10)]

Asking for the backend is what imports this module, and importing it is what puts
:code:`MultiVector.__torch_function__` on a multivector to make multivectors
work with :code:`torch`.

A torch function is handed :code:`mv.values()`, and its result becomes the coefficients of the
multivector that comes back. That is the whole of it, and it is enough because a multivector over
an array of shape :code:`(blades, ..., channels)` has shape :code:`(..., channels)`: the blade axis
is a *leading* axis, and everything :code:`torch.nn` is built out of treats leading axes as batch
axes. So a :code:`torch.nn.Linear` with weights of :code:`(channels, channels_out)` acts on the
coefficients of every blade at once and cannot reach the blades, and nothing here needs to keep a
list of which functions those are.

To address the axes of the multivector itself, :code:`einops` speaks multivector: use
:code:`einops.reduce`, :code:`einops.rearrange`, :code:`einops.einsum` and :code:`einops.pack` on
it, where the patterns refer to :code:`mv.shape` and the blade axis stays out of it. This module
already registers :class:`~kingdon.einops_backend.KingdonBackend` for you if :code:`einops` is installed.

The exception to all of the above is one rule: **if a multivector has an operation of that name,
torch's name means the multivector's.** :code:`_OPERATIONS` is that rule as a table, and it holds on
whichever side of an operator the multivector sits -- :code:`tensor | mv` is the inner product just
as :code:`mv | tensor` is. So :code:`torch.mul` is the geometric product like :code:`*` is,
:code:`torch.matmul` is the projection like :code:`@` is, and :code:`torch.exp` is the exponential
of the multivector like :meth:`~kingdon.multivector.MultiVector.exp` is. A name a multivector does
not have is handed the coefficients as ever, so :code:`torch.relu(mv)` is the relu of every one of
them, and :code:`mv.values()` is there when the coefficients are what you mean::

    >>> torch.exp(bivector)            # a rotor
    >>> torch.exp(bivector.values())   # the exponential of every coefficient
"""
from __future__ import annotations

import inspect

import sympy
import sympy.printing.pytorch
import torch

from kingdon.multivector import MultiVector

try:
    # Whoever works with torch wants einops.reduce and friends too, and should not have to find a
    # second import for them. Nothing here needs einops, so its absence is not an error.
    import kingdon.einops_backend  # noqa: F401
except ImportError:  # pragma: no cover
    pass


class TorchPrinter(sympy.printing.pytorch.TorchPrinter):
    """
    Prints an operator whose codegen_symbolcls is a sympy symbol, and which can therefore call sympy's functions -- :code:`erf`, say -- as torch code.
    A constant is printed as a number, since :code:`torch.sqrt(2)` wants a tensor.
    """
    namespace = {'torch': torch}

    def _print(self, expr, **kwargs):
        if isinstance(expr, sympy.Basic) and expr.is_number and not expr.is_Integer:
            return repr(float(expr))
        return super()._print(expr, **kwargs)


def values_asarray(values):
    """
    The coefficients of a multivector as a single tensor whose first axis is the blade axis. This
    is what :code:`Algebra(..., backend='torch')` sets as its
    :code:`values_asarray`; pass it yourself if you want it without the rest of the backend::

        >>> alg = Algebra.fromname('3DPGA', values_asarray=values_asarray)

    Coefficients that do not already agree are broadcast against each other, so that the plain
    python numbers a type's layout contributes -- the :code:`1.0` that a normalized :code:`Point`
    carries on :code:`e123`, say -- do not stop a multivector from having a shape. Which of the two
    it is gets decided by inspecting the values, not by catching what :code:`torch.stack` raises:
    an exception out of a torch call is a graph break, and one here would break the graph of every
    :code:`torch.compile` that traces a multivector expression.

    `values` that hold no tensor at all are returned untouched, leaving them to kingdon's default
    of a plain list. That is not an edge case: :class:`~kingdon.algebra.BladeDict` builds every
    basis blade of the algebra out of a plain :code:`1`, so an algebra cannot even be constructed
    without it. Symbolic multivectors never get here, since every path that makes one passes
    :code:`raw=True` to :meth:`~kingdon.multivector.MultiVector.fromkeysvalues`.
    """
    if not isinstance(values, (list, tuple)):
        return values  # Already one tensor, which is what this is for in the first place.
    like = next((v for v in values if isinstance(v, torch.Tensor)), None)
    if like is None:
        return values  # No tensor here at all: the plain 1s that alg.blades is built from.
    if all(isinstance(v, torch.Tensor) and v.shape == like.shape and v.device == like.device
           for v in values):
        return torch.stack(values)  # The common case; note that torch promotes the dtypes itself.
    # Different shapes, or a plain number among the tensors.
    return torch.stack(torch.broadcast_tensors(
        *(torch.as_tensor(v, dtype=like.dtype, device=like.device) for v in values)))


#: What a torch name means for a multivector, as :code:`{torch name: the name kingdon has for it}`.
#: The operators are the ones the two spell differently, and the last four are torch's dunders,
#: since it has no name of its own for :code:`|` and friends. The rest are asked of
#: :class:`~kingdon.multivector.MultiVector`, so that the rule follows kingdon rather than a list
#: here going stale: every operation it has that torch has a name for. Today that is exp, norm and
#: sqrt, and also layout, which torch spells as a class, so no call of it ever arrives here.
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
    Register multivector `types` with :code:`torch.utils._pytree`, so that :code:`torch.export` can
    take and give back a multivector rather than refuse it as a type it does not know how to
    flatten.

    :code:`torch.compile` needs none of this: dynamo traces straight through a multivector as the
    plain python object it is, and reaches one graph with no breaks either way. Export is the one
    that flattens whatever crosses its boundary, and it wants the keyed flatten besides.

    The coefficients are the only child, since they are the tensor to trace; the type, the algebra
    and the keys are static context, which is what makes the sparsity pattern of a multivector a
    compile time constant that the graph specializes on. Export hashes that context, which is why
    :class:`~kingdon.algebra.Algebra` defines :code:`Algebra.__hash__`. Types are
    registered per algebra, because :class:`~kingdon.algebra.Algebra` generates classes of its own
    for the layouts it is given.

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


# ---------------------------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------------------------

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
