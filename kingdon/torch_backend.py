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
:meth:`~kingdon.multivector.MultiVector.__torch_function__` on a multivector to make multivectors
work with :code:`torch`.

A torch function is handed :code:`mv.values()`, and its result becomes the coefficients of the
multivector that comes back. That is the whole of it, and it is enough because a multivector over
an array of shape :code:`(blades, ..., channels)` has shape :code:`(..., channels)`: the blade axis
is a *leading* axis, and everything :code:`torch.nn` is built out of treats leading axes as batch
axes. So a :class:`~torch.nn.Linear` with weights of :code:`(channels, channels_out)` acts on the
coefficients of every blade at once and cannot reach the blades, and nothing here needs to keep a
list of which functions those are.

To address the axes of the multivector itself, :code:`einops` speaks multivector: use
:func:`einops.reduce`, :func:`einops.rearrange`, :func:`einops.einsum` and :func:`einops.pack` on
it, where the patterns refer to :code:`mv.shape` and the blade axis stays out of it. This module
already registers :class:`~kingdon.einops_backend.KingdonBackend` for you if :code:`einops` is installed.

The exception to all of the above are the operators. However torch spells one, it is handed to the
algebra, so that :func:`torch.mul` is the geometric product like :code:`*` is, :func:`torch.matmul`
is the projection like :code:`@` is, and so on down :code:`+ - * / @ | ^ & >>`. This holds on
whichever side of the operator the multivector sits: :code:`tensor | mv` is the inner product just
as :code:`mv | tensor` is.

No other name in the torch namespace means geometric algebra, so :code:`torch.exp(mv)`
exponentiates the coefficients while :code:`mv.exp()` is the exponential of the multivector.
"""
from __future__ import annotations

import torch

from kingdon.multivector import MultiVector

try:
    # Whoever works with torch wants einops.reduce and friends too, and should not have to find a
    # second import for them. Nothing here needs einops, so its absence is not an error.
    import kingdon.einops_backend  # noqa: F401
except ImportError:  # pragma: no cover
    pass


def values_asarray(values):
    """
    The coefficients of a multivector as a single tensor whose first axis is the blade axis. This
    is what :code:`Algebra(..., backend='torch')` sets as its
    :code:`values_asarray`; pass it yourself if you want it without the rest of the backend::

        >>> alg = Algebra.fromname('3DPGA', values_asarray=values_asarray)

    Coefficients that do not already agree are broadcast against each other, so that the plain
    python numbers a type's layout contributes -- the :code:`1.0` that a normalized :code:`Point`
    carries on :code:`e123`, say -- do not stop a multivector from having a shape. Which of the two
    it is gets decided by inspecting the values, not by catching what :func:`torch.stack` raises:
    an exception out of a torch call is a graph break, and one here would break the graph of every
    :func:`torch.compile` that traces a multivector expression.

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


def _operator(name):
    """ Hand `name` to the algebra, so that torch.mul is the geometric product. """
    def handler(input, other=None, *, alpha=1):
        mv = input if isinstance(input, MultiVector) else other
        operator = getattr(mv.algebra, name)
        return operator(input) if other is None else \
            operator(input, other if alpha == 1 else alpha * other)
    return handler


#: How torch spells each operator that a multivector defines for itself, as
#: :code:`{torch name: algebra operator}`. Torch has no name of its own for :code:`|` and the
#: others, so those are its dunders, which exist on :class:`torch.Tensor` alone.
_OPERATORS = {'add': 'add', 'sub': 'sub', 'subtract': 'sub', 'mul': 'gp', 'multiply': 'gp',
              'div': 'div', 'divide': 'div', 'true_divide': 'div', 'neg': 'neg', 'negative': 'neg',
              'matmul': 'proj', '__or__': 'ip', '__xor__': 'op', '__and__': 'rp',
              '__rshift__': 'sw'}

#: The functions with a handler of their own, as :code:`{torch function: handler}`.
_HANDLED = {func: _operator(operator)
            for name, operator in _OPERATORS.items() for namespace in (torch, torch.Tensor)
            if (func := getattr(namespace, name, None)) is not None}


def torch_function(func, types, args=(), kwargs=None):
    """
    Implementation of :meth:`MultiVector.__torch_function__
    <kingdon.multivector.MultiVector.__torch_function__>`.
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
    Implementation of the torch half of :meth:`MultiVector.__getattr__
    <kingdon.multivector.MultiVector.__getattr__>`, called for an attribute that is not a basis
    blade once torch has been imported.

    A tensor method never reaches :meth:`~kingdon.multivector.MultiVector.__torch_function__`,
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
    Register multivector `types` with :mod:`torch.utils._pytree`, so that :func:`torch.compile`
    can trace a function that takes or returns a multivector instead of breaking its graph on one.

    The coefficients are the only child, since they are the tensor to trace; the type, the algebra
    and the keys are static context, which is what makes the sparsity pattern of a multivector a
    compile time constant that the graph specializes on. Types are registered per algebra, because
    :class:`~kingdon.algebra.Algebra` generates classes of its own for the layouts it is given.

    :param types: multivector classes to register. Registering one twice is a no-op.
    """
    from torch.utils._pytree import register_pytree_node

    for cls in types:
        if cls in _pytree_registered:
            continue
        register_pytree_node(
            cls,
            lambda mv: ([mv._values], (type(mv), mv.algebra, mv._keys)),
            lambda values, context: context[0].fromkeysvalues(context[1], context[2], next(iter(values))),
        )
        _pytree_registered.add(cls)


# ---------------------------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------------------------

_kingdon_getattr = MultiVector.__getattr__


def _getattr(self, name):
    """ :meth:`MultiVector.__getattr__`, with torch behind the basis blades. """
    try:
        return _kingdon_getattr(self, name)
    except AttributeError:
        # Private names are left alone, so copy and pickle cannot recurse into a half built mv.
        if name.startswith('_'):
            raise
        return torch_getattr(self, name)


def _torch_function(cls, func, types, args=(), kwargs=None):
    """ :meth:`MultiVector.__torch_function__`; the work is in :func:`torch_function`. """
    return torch_function(func, types, args, kwargs or {})


# Importing this module is what opts a multivector in to torch, and an
# :class:`~kingdon.algebra.Algebra` with :code:`backend='torch'` is what imports it. Until then
# :class:`~kingdon.multivector.MultiVector` has no :code:`__torch_function__` at all, so torch does
# not treat it as a type that overrides anything and :code:`tensor * mv` falls through to
# :meth:`~kingdon.multivector.MultiVector.__rmul__` exactly as it does without torch installed.
MultiVector.__getattr__ = _getattr
MultiVector.__torch_function__ = classmethod(_torch_function)
