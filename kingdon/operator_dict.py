from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple
import string

from sympy import Symbol, Expr, simplify

from kingdon.multivector import MultiVector
from kingdon.codegen import do_codegen, do_compile
from kingdon.taperecorder import TapeRecorder


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
            mvs = [self.algebra.multivector(name=name, keys=keys, symbolcls=self.algebra.codegen_symbolcls)
                   for name, keys in zip(string.ascii_lowercase, keys_in)]
            keys_out, func = do_codegen(self.codegen, *mvs)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = (keys_out, func)
        return self.operator_dict[keys_in]

    def __contains__(self, keys_in: Tuple[Tuple[int]]):
        return keys_in in self.operator_dict

    def __iter__(self):
        return iter(self.operator_dict)

    def filter(self, keys_out, values_out):
        """ For given keys and values, keep only symbolically non-zero elements. """
        keysvalues = tuple((k, simpv) for k, v in zip(keys_out, values_out) if (simpv := self.algebra.simp_func(v)))
        keys, values = zip(*keysvalues) if keysvalues else (tuple(), list())
        return keys, list(values)

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
        keys_out, func = self[keys_in]
        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic or not mvs[0].algebra.wrapper:
            values_out = func(*values_in)
        else:
            values_out = self.algebra.numspace[func.__name__](*values_in)

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)

    def _call_binary(self, mv1, mv2):
        """ Specialization for binary operators. """
        # Call until no longer callable.
        while isinstance(mv1, Callable) and not isinstance(mv1, MultiVector):
            mv1 = mv1()
        while isinstance(mv2, Callable) and not isinstance(mv2, MultiVector):
            mv2 = mv2()

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mv1 = mv1 if isinstance(mv1, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv1])
        mv2 = mv2 if isinstance(mv2, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv2])
        # Check is written to be fast, not readable. Typically, the first check is true.
        if not (mv1.algebra is mv2.algebra or mv1.algebra == mv2.algebra):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_out, func = self[mv1.keys(), mv2.keys()]
        issymbolic = (mv1.issymbolic or mv2.issymbolic)
        if issymbolic or not mv1.algebra.wrapper:
            values_out = func(mv1.values(), mv2.values())
        else:
            values_out = self.algebra.numspace[func.__name__](mv1.values(), mv2.values())

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


class UnaryOperatorDict(OperatorDict):
    """
    Specialization of OperatorDict for unary operators. In the
    case of unary operators, we can do away with all of the overhead that is necessary for
    operators that act on multiple multivectors.
    """
    def __getitem__(self, keys_in: Tuple[Tuple[int]]):
        if keys_in not in self.operator_dict:
            mv = self.algebra.multivector(name='a', keys=keys_in, symbolcls=self.algebra.codegen_symbolcls)
            keys_out, func = do_codegen(self.codegen, mv)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = (keys_out, func)
        return self.operator_dict[keys_in]

    def __call__(self, mv):
        keys_out, func = self[mv.keys()]

        issymbolic = mv.issymbolic
        if issymbolic or not mv.algebra.wrapper:
            values_out = func(mv.values())
        else:
            values_out = self.algebra.numspace[func.__name__](mv.values())

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


class Registry(OperatorDict):
    def __getitem__(self, keys_in: Tuple[Tuple[int]]):
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            tapes = [TapeRecorder(algebra=self.algebra, expr=name, keys=keys)
                     for name, keys in zip(string.ascii_lowercase, keys_in)]
            keys_out, func = do_compile(self.codegen, *tapes)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = (keys_out, func)
        return self.operator_dict[keys_in]

    def __call__(self, *mvs):
        mvs = list(mvs)
        for i in range(len(mvs)):
            mv = mvs[i]
            # Call until no longer callable.
            while isinstance(mv, Callable) and not isinstance(mv, MultiVector):
                mv = mv()
            mvs[i] = mv

        if all(isinstance(mv, TapeRecorder) for mv in mvs):
            keys_in = tuple(mv.keys() for mv in mvs)
            keys_out, func = self[keys_in]
            expr = f"{func.__name__}({', '.join(mv.expr for mv in mvs)})"
            return TapeRecorder(self.algebra, keys=keys_out, expr=expr)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = tuple(mv.keys() for mv in mvs)
        values_in = tuple(mv.values() for mv in mvs)
        keys_out, func = self[keys_in]

        if not mvs[0].algebra.wrapper:
            values_out = func(*values_in)
        else:
            values_out = self.algebra.numspace[func.__name__](*values_in)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)
