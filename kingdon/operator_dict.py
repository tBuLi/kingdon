from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple
import string

from numba import njit
from sympy import Symbol, Expr, simplify

from kingdon.multivector import MultiVector


class AlgebraError(Exception):
    pass


@dataclass
class OperatorDict(Mapping):
    """
    A dict-like object which performs codegen of a particular operator,
    and caches the result for future use. For example, to generate the geometric product,
    we create an OperatorDict as follows::

        alg = Algebra(3, 0, 1)
        gp = OperatorDict('gp', codegen=codegen_gp, algebra=alg)

    Here, :code:`codegen_gp` is a function that outputs the keys of the result, and a callable that
    produces the corresponding values. See :class:`~kingdon.codegen.CodegenOutput` for more info.
    """
    name: str
    codegen: Callable
    algebra: "Algebra"
    operator_dict: dict = field(default_factory=dict, init=False)

    def __len__(self):
        return len(self.operator_dict)

    def __getitem__(self, keys_in: Tuple[Tuple[int]]):
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            mvs = tuple(
                MultiVector.fromkeysvalues(
                    self.algebra, keys=keys,
                    values=tuple(Symbol(f'{name}{self.algebra.bin2canon[ek][1:]}') for ek in keys))
                for name, keys in zip(string.ascii_lowercase, keys_in)
            )
            keys_out, func = self.codegen(*mvs)
            self.operator_dict[keys_in] = (keys_out, func, njit(func))
        return self.operator_dict[keys_in]

    def __iter__(self):
        return iter(self.operator_dict)

    def __call__(self, *mvs):
        if len(mvs) == 2:
            return self._call_binary(*mvs)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = tuple(mv.keys() for mv in mvs)
        values_in = tuple(mv.values() for mv in mvs)
        keys_out, func, numba_func = self[keys_in]
        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic or not mvs[0].algebra.numba:
            values_out = func(*values_in)
        else:
            values_out = numba_func(*values_in)

        if issymbolic and self.algebra.simplify:
            # Keep only symbolically non-zero elements.
            keysvalues = tuple(filter(
                lambda kv: True if not isinstance(kv[1], Expr) else simplify(kv[1]),
                zip(keys_out, values_out)
            ))
            keys_out, values_out = zip(*keysvalues) if keysvalues else (tuple(), tuple())

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)

    def _call_binary(self, mv1, mv2):
        """ Specialization for binary operators. """
        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mv1 = mv1 if isinstance(mv1, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv1,))
        mv2 = mv2 if isinstance(mv2, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv2,))
        # Check is written to be fast, not readable. Typically, the first check is true.
        if not (mv1.algebra is mv2.algebra or mv1.algebra == mv2.algebra):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_out, func, numba_func = self[mv1.keys(), mv2.keys()]
        issymbolic = (mv1.issymbolic or mv2.issymbolic)
        if issymbolic or not mv1.algebra.numba:
            values_out = func(mv1.values(), mv2.values())
        else:
            values_out = numba_func(mv1.values(), mv2.values())

        if issymbolic and self.algebra.simplify:
            # Keep only symbolically non-zero elements.
            keysvalues = tuple(filter(
                lambda kv: True if not isinstance(kv[1], Expr) else simplify(kv[1]),
                zip(keys_out, values_out)
            ))
            keys_out, values_out = zip(*keysvalues) if keysvalues else (tuple(), tuple())

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


class UnaryOperatorDict(OperatorDict):
    """
    Specialization of OperatorDict for unary operators. In the
    case of unary operators, we can do away with all of the overhead that is necessary for
    operators that act on multiple multivectors.
    """
    def __getitem__(self, keys_in: Tuple[Tuple[int]]):
        if keys_in not in self.operator_dict:
            vals = tuple(Symbol(f'a{self.algebra.bin2canon[ek][1:]}') for ek in keys_in)
            mv = MultiVector.fromkeysvalues(self.algebra, keys=keys_in, values=vals)
            keys_out, func = self.codegen(mv)
            self.operator_dict[keys_in] = (keys_out, func, njit(func))
        return self.operator_dict[keys_in]

    def __call__(self, mv):
        keys_out, func, numba_func = self[mv.keys()]

        issymbolic = mv.issymbolic
        if issymbolic or not mv.algebra.numba:
            values_out = func(mv.values())
        else:
            values_out = numba_func(mv.values())

        if issymbolic and self.algebra.simplify:
            # Keep only symbolically non-zero elements.
            keysvalues = tuple(filter(
                lambda kv: True if not isinstance(kv[1], Expr) else simplify(kv[1]),
                zip(keys_out, values_out)
            ))
            keys_out, values_out = zip(*keysvalues) if keysvalues else (tuple(), tuple())

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)
