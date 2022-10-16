from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable
import string

from sympy import Symbol, Expr, simplify

from kingdon.multivector import MultiVector


class AlgebraError(Exception):
    pass


@dataclass
class OperatorDict(Mapping):
    name: str
    codegen: Callable
    algebra: "Algebra"
    operator_dict: dict = field(default_factory=dict, init=False)

    def __len__(self):
        return len(self.operator_dict)

    def __getitem__(self, keys_in):
        if keys_in not in self.operator_dict:
            mvs = []
            for name, keys in zip(string.ascii_lowercase, keys_in):
                vals = tuple(Symbol(f'{name}{self.algebra.bin2canon[ek][1:]}') for ek in keys)
                mvs.append(MultiVector.fromkeysvalues(self.algebra, keys=keys, values=vals))
            self.operator_dict[keys_in] = self.codegen(*mvs)
        else:
            self.operator_dict[keys_in]
        return self.operator_dict[keys_in]

    def __iter__(self):
        return iter(self.operator_dict)

    def __call__(self, *mvs):
        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")
        keys_in = tuple(mv.keys() for mv in mvs)
        values_in = tuple(mv.values() for mv in mvs)
        keys_out, func = self[keys_in]
        values = func(*values_in)

        if self.algebra.simplify and any(mv.issymbolic for mv in mvs):
            # Keep only symbolically non-zero elements.
            keysvalues = tuple(filter(
                lambda kv: True if not isinstance(kv[1], Expr) else simplify(kv[1]),
                zip(keys_out, values)
            ))
            keys_out, values = zip(*keysvalues) if keysvalues else (tuple(), tuple())

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values)
