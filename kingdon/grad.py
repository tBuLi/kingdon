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

    def __post_init__(self):
        self.algebra = self.coordinate_mv.algebra

    def __mul__(self, other: Callable) -> Callable:
        @wraps(other)
        def grad_other(*args, **kwargs):
            res = self.algebra.multivector()
            for k in self.coordinate_mv.keys():
                v_prime = self.algebra.multivector(keys=(k,), values=(EPSILON,))
                blade = self.algebra.blades[self.algebra.bin2canon[k]]
                partial = other(self.coordinate_mv + v_prime)
                if not isinstance(partial, MultiVector) or partial.algebra != self.algebra:
                    partial = self.algebra.scalar(e=partial)
                res += blade.inv() * partial.filter(lambda v: hasattr(v, 'e0')).map(lambda v: v.e0)
            return res


        grad_other.__name__ = f"{other.__name__}_grad"
        grad_other.__doc__ = f"Gradient of {other.__name__}."
        return grad_other
