from dataclasses import dataclass, field
from kingdon.multivector import MultiVector
from kingdon.algebra import Algebra
from typing import Callable
from functools import wraps

DUAL_ALGEBRA = Algebra(r=1)	
EPSILON = DUAL_ALGEBRA.vector(e0=1)

@dataclass
class Grad:
    coordinate_mv: MultiVector
    algebra: "Algebra" = field(init=False)
    partial_derivative: Callable | None = field(default=None)

    def __post_init__(self):
        self.algebra = self.coordinate_mv.algebra
        if self.partial_derivative is None:
            self.partial_derivative = partial_derivative

    def __mul__(self, other: Callable) -> Callable:
        return make_grad_func(other, self.coordinate_mv, self.partial_derivative, MultiVector.__mul__)

    def __or__(self, other: Callable) -> Callable:
        return make_grad_func(other, self.coordinate_mv, self.partial_derivative, MultiVector.__or__)

    def __xor__(self, other: Callable) -> Callable:
        return make_grad_func(other, self.coordinate_mv, self.partial_derivative, MultiVector.__xor__)

    def __and__(self, other: Callable) -> Callable:
        return make_grad_func(other, self.coordinate_mv, self.partial_derivative, MultiVector.__and__)


def partial_derivative(other: Callable, coordinate_mv: MultiVector, k: int) -> MultiVector:
    alg = coordinate_mv.algebra
    v_prime = alg.multivector(keys=(k,), values=(EPSILON,))
    partial = other(coordinate_mv + v_prime)
    if not isinstance(partial, MultiVector) or partial.algebra != alg:
        partial = alg.scalar(e=partial)
    return partial.filter(lambda v: hasattr(v, 'e0')).map(lambda v: v.e0)


def partial_derivative_sympy(other, coordinate_mv, k):
    var = getattr(coordinate_mv, coordinate_mv.algebra.bin2canon[k])
    return other.map(lambda v: v.diff(var))


def make_grad_func(other: Callable, coordinate_mv: MultiVector, partial_derivative: Callable, operator: Callable) -> Callable:
    """ Make a gradient function for the given binary operator. """
    alg = coordinate_mv.algebra

    if isinstance(other, MultiVector) and other.issymbolic:
        # If symbolic, do the derivation symbolically.
        res = alg.multivector()
        for k in coordinate_mv.keys():
            partial = partial_derivative_sympy(other, coordinate_mv, k)
            blade = alg.blades[alg.bin2canon[k]]
            res += operator(blade.inv(), partial)  # TODO: Inv should be replaced by reciprocal.
        return res

    @wraps(other)
    def grad_other(*args, **kwargs):
        # TODO: Handle multivariable functions.
        res = alg.multivector()
        for k in coordinate_mv.keys():
            partial = partial_derivative(other, coordinate_mv, k)
            blade = alg.blades[alg.bin2canon[k]]
            res += operator(blade.inv(), partial)  # TODO: Inv should be replaced by reciprocal.
        return res

    grad_other.__name__ = f"{other.__name__}_grad"
    grad_other.__doc__ = f"Gradient of {other.__name__}."
    return grad_other
