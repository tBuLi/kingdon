from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple, NamedTuple
from functools import wraps, cached_property
import inspect
import string

from sympy import Symbol, Expr, simplify

from kingdon.multivector import MultiVector
from kingdon.codegen import do_codegen, do_compile
from kingdon.taperecorder import TapeRecorder
from kingdon.polynomial import RationalPolynomial


class AlgebraError(Exception):
    pass


def resolve_and_expand(func):
    """
    Decorator which makes :code:`func` compatible function over MVs compatible with the broader ganja.js style
    broadcasting rules:
    - binary and unary operators can be applied to lists & tuples, e.g. :code:`x * [y, z]` or :code:`alg.gp([y, z], x)`.
    - binary and unary operators can be applied to functions without any arguments, e.g. :code:`x * lambda: y * z` or
    :code:`alg.gp(lambda: y * z, x)`.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    is_method = len(params) > 0 and params[0].name == "self"

    if is_method:
        @wraps(func)
        def wrapper(self, *mvs):
            mvs = list(mvs)
            for i in range(len(mvs)):
                mv = mvs[i]
                # Call until no longer callable.
                while isinstance(mv, Callable) and not isinstance(mv, MultiVector):
                    mv = mv()
                mvs[i] = mv

            for i in range(len(mvs)):
                mv = mvs[i]
                if isinstance(mv, (tuple, list)):
                    return type(mv)(wrapper(self, *(mvs[:i] + [x] + mvs[i + 1:])) for x in mv)

            return func(self, *mvs)
        return wrapper

    @wraps(func)
    def wrapper(*mvs):
        mvs = list(mvs)
        for i in range(len(mvs)):
            mv = mvs[i]
            # Call until no longer callable.
            while isinstance(mv, Callable) and not isinstance(mv, MultiVector):
                mv = mv()
            mvs[i] = mv

        for i in range(len(mvs)):
            mv = mvs[i]
            if isinstance(mv, (tuple, list)):
                return type(mv)(wrapper(*(mvs[:i] + [x] + mvs[i + 1:])) for x in mv)

        return func(*mvs)

    return wrapper


def do_operation(*mvs, codegen, algebra) -> MultiVector:
    """
    This function just does the operation directly on the MV's, no codegen is performed.
    This is used for large algebras, where codegen is too costly.
    The result is the multivector resulting from :code:`codegen(*mvs)`.
    """
    mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(algebra, (0,), (mv,))
           for mv in mvs]
    if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
        raise AlgebraError("Cannot multiply elements of different algebra's.")

    res = codegen(*mvs)
    if isinstance(res, MultiVector):
        return res
    elif isinstance(res, dict):
        # TODO: Can this sort be done without canon2bin?
        res = {bin: res[bin] if isinstance(res, dict) else getattr(res, canon)
               for canon, bin in algebra.canon2bin.items() if bin in res.keys()}
        if not res:
            return MultiVector.fromkeysvalues(algebra, tuple(), [])
        keys, values = zip(*res.items())
        return MultiVector.fromkeysvalues(algebra, keys, list(values)).filter()
    else:
        # TODO: there is probably something better than raising an error.
        raise NotImplementedError(type(res))


class OperatorDictOutput(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    """
    keys_out: Tuple[int]
    func: Callable


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
    codegen_symbolcls: Callable = field(default=RationalPolynomial.fromname, repr=False)

    def __post_init__(self):
        if self.algebra.codegen_symbolcls is not None:
            # If the user forces a different codegen_symbolcls then give them what they want.
            self.codegen_symbolcls = self.algebra.codegen_symbolcls

    def __len__(self):
        return len(self.operator_dict)

    def _make_symbolic_mv(self, name, MVType, keys, shape):
        MVType, depth = MVType if isinstance(MVType, tuple) else (MVType, 0)
        depth = shape[1] if depth is None else depth
        names = list(f'{name}{self.algebra.bin2canon[k][1:]}' for k in keys)
        if not depth:
            return MVType.fromkeysvalues(self.algebra, keys, list(self.codegen_symbolcls(name) for name in names))
        mv_values = [[self.codegen_symbolcls(f'{name}_{j}') for j in range(depth)] for name in names]
        return MVType.fromkeysvalues(self.algebra, keys, mv_values)

    def make_symbolic_mvs(self, keys_in: Tuple[Tuple[int]], shapes_in: Tuple[Tuple[int]]) -> tuple[MultiVector]:
        return tuple(self._make_symbolic_mv(name, MVType, keys, shape) for (name, MVType), keys, shape in zip(self.codegen_input_types.items(), keys_in, shapes_in))

    def __getitem__(self, mvs: Tuple[MultiVector]):
        keys_in = tuple(mv.keys() for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            mvs = self.make_symbolic_mvs(keys_in, shapes_in)
            keys_out, func = do_codegen(self.codegen, *mvs)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = OperatorDictOutput(keys_out, func)
        return self.operator_dict[keys_in]

    def __contains__(self, mvs: Tuple[MultiVector]):
        keys_in = tuple(mv.keys() for mv in mvs)
        return keys_in in self.operator_dict

    def __iter__(self):
        return iter(self.operator_dict)

    def filter(self, keys_out, values_out):
        """ For given keys and values, keep only symbolically non-zero elements. """
        keysvalues = tuple((k, simpv) for k, v in zip(keys_out, values_out) if (simpv := self.algebra.simp_func(v)))
        keys, values = zip(*keysvalues) if keysvalues else (tuple(), list())
        return keys, list(values)

    @cached_property
    def codegen_input_types(self):
        return {name: MultiVector if p.annotation == inspect.Parameter.empty else p.annotation
                for name, p in inspect.signature(self.codegen).parameters.items()}

    @cached_property
    def codegen_output_type(self):
        return_annotation = inspect.signature(self.codegen).return_annotation
        return MultiVector if return_annotation == inspect.Signature.empty else return_annotation

    def _sanitize_mvs(self, mvs: tuple[MultiVector]):
        """ Make sure all inputs are multivectors. If an input is not, assume its scalar."""
        if len(mvs) == 1:
            mv = mvs[0]
            mv = mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv])
            return (mv,)
        if len(mvs) == 2:
            mv1, mv2 = mvs
            mv1 = mv1 if isinstance(mv1, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv1])
            mv2 = mv2 if isinstance(mv2, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv2])
            mvs = (mv1, mv2)
        else:
            mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), [mv])
                   for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")
        return mvs

    @resolve_and_expand
    def __call__(self, *mvs):
        mvs = self._sanitize_mvs(mvs)
        if len(mvs) == 2:
            return self._call_binary(*mvs)

        keys_out, func = self[mvs]
        values_in = tuple(mv.values() for mv in mvs)
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
        keys_out, func = self[mv1, mv2]
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
    def __contains__(self, mv: MultiVector):
        keys_in = mv.keys()
        return keys_in in self.operator_dict

    def __getitem__(self, mv: MultiVector):
        keys_in = mv.keys()
        shape_in = mv.shape
        if keys_in not in self.operator_dict:
            mv = self.make_symbolic_mvs((keys_in,), (shape_in,))[0]
            keys_out, func = do_codegen(self.codegen, mv)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = OperatorDictOutput(keys_out, func)
        return self.operator_dict[keys_in]

    @resolve_and_expand
    def __call__(self, mv):
        mv = self._sanitize_mvs((mv,))[0]
        keys_out, func = self[mv]

        issymbolic = mv.issymbolic
        if issymbolic or not mv.algebra.wrapper:
            values_out = func(mv.values())
        else:
            values_out = self.algebra.numspace[func.__name__](mv.values())

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


class Registry(OperatorDict):
    def __getitem__(self, mvs: Tuple[MultiVector]):
        keys_in = tuple(mv.keys() for mv in mvs)
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            tapes = [TapeRecorder(algebra=self.algebra, expr=name, keys=keys)
                     for name, keys in zip(string.ascii_lowercase, keys_in)]
            keys_out, func = do_compile(self.codegen, *tapes)
            self.algebra.numspace[func.__name__] = self.algebra.wrapper(func) if self.algebra.wrapper else func
            self.operator_dict[keys_in] = OperatorDictOutput(keys_out, func)
        return self.operator_dict[keys_in]

    @resolve_and_expand
    def __call__(self, *mvs):
        if all(isinstance(mv, TapeRecorder) for mv in mvs):
            keys_out, func = self[mvs]
            expr = f"{func.__name__}({', '.join(mv.expr for mv in mvs)})"
            return TapeRecorder(self.algebra, keys=keys_out, expr=expr)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        values_in = tuple(mv.values() for mv in mvs)
        keys_out, func = self[mvs]

        if not mvs[0].algebra.wrapper:
            values_out = func(*values_in)
        else:
            values_out = self.algebra.numspace[func.__name__](*values_in)

        return MultiVector.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)
