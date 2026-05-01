from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple, NamedTuple, Type
from functools import wraps, cached_property
import inspect
import string

from sympy import Symbol, Expr, simplify
from sympy.printing.lambdarepr import LambdaPrinter

from kingdon.multivector import MultiVector, stack
from kingdon.codegen import do_codegen, do_compile, KingdonPrinter
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
        def wrapper(self, *mvs, **kwargs):
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
                    return type(mv)(wrapper(self, *(mvs[:i] + [x] + mvs[i + 1:]), **kwargs) for x in mv)

            return func(self, *mvs, **kwargs)
        return wrapper

    @wraps(func)
    def wrapper(*mvs, **kwargs):
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
                return type(mv)(wrapper(*(mvs[:i] + [x] + mvs[i + 1:]), **kwargs) for x in mv)

        return func(*mvs, **kwargs)

    return wrapper


def do_operation(*mvs, codegen, algebra, MVType=MultiVector) -> MultiVector:
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
            return MVType.fromkeysvalues(algebra, tuple(), [])
        keys, values = zip(*res.items())
        return MVType.fromkeysvalues(algebra, keys, list(values)).filter()
    else:
        # TODO: there is probably something better than raising an error.
        raise NotImplementedError(type(res))


class OperatorDictOutput(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    :param wrapped_func: decorated func if a wrapper was provided, else identical to func.
    :param MVType: type of the output multivector. Defaults to :code:`MultiVector`.
    """
    keys_out: Tuple[int]
    func: Callable
    wrapped_func: Callable | None = None
    MVType: Type = MultiVector


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
    printer: LambdaPrinter = field(default=None, repr=False)
    func_printer: KingdonPrinter = field(default=None, repr=False)
    wrapper: Callable = field(default=None, repr=False)

    def __post_init__(self):
         # If the user forces a different codegen settings for this operator then give them what they want.
        if not self.codegen_symbolcls and self.algebra.codegen_symbolcls:
            self.codegen_symbolcls = self.algebra.codegen_symbolcls
        if not self.printer and self.algebra.printer:
            self.printer = self.algebra.printer
        if not self.func_printer and self.algebra.func_printer:
            self.func_printer = self.algebra.func_printer
        if not self.wrapper and self.algebra.wrapper:
            self.wrapper = self.algebra.wrapper

    def __len__(self):
        return len(self.operator_dict)

    def _make_symbolic_mv(self, name, MVType, keys, shape):
        MVType, depth = MVType if isinstance(MVType, tuple) else (MVType, 0)  # Can come from the type-hint.
        depth = shape[1] if depth is None else depth
        if not depth:
            return MVType.fromname(self.algebra, name, keys, symbolcls=self.codegen_symbolcls)
        return stack([MVType.fromname(self.algebra, f'{name}_{k}', keys, symbolcls=self.codegen_symbolcls) 
                      for k in range(depth)])

    def make_symbolic_mvs(self, types_in: Tuple[Type, Tuple[int]], shapes_in: Tuple[Tuple[int]]) -> tuple[MultiVector]:
        return tuple(
            self._make_symbolic_mv(name, MVType, keys, shape) 
            for (name, MVTypeHint), (MVType, keys), shape in zip(self.codegen_input_types.items(), types_in, shapes_in)
        )

    def __getitem__(self, mvs: Tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if types_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            mvs = self.make_symbolic_mvs(types_in, shapes_in)
            keys_out, func, MVType = do_codegen(self.codegen, *mvs, printer=self.printer, func_printer=self.func_printer, type_patterns=self.algebra.type_patterns.get(self.name, {}))
            self.algebra.numspace[func.__name__] = wrapped_func = self.wrapper(func) if self.wrapper else func
            self.operator_dict[types_in] = OperatorDictOutput(keys_out, func, wrapped_func, MVType)
        return self.operator_dict[types_in]

    def __contains__(self, mvs: Tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        return types_in in self.operator_dict

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
        """
        Make sure all inputs match their type annotations.
        If an input is expected to be a MultiVector but is not, cast it to a scalar.
        # TODO: make sure that if the typehint was trivial, then we do not want to pay the price
        #   of fixing an interesting typehint.
        """
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

        keys_out, func, wrapped_func, _MVType = self[mvs]
        MVType = _MVType or MultiVector
        values_in = tuple(mv.values() for mv in mvs)
        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic:
            values_out = func(*values_in)
        else:
            values_out = wrapped_func(*values_in)

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)
            output_mv_idx = getattr(func, 'output_mv_idx', None)
            if output_mv_idx is not None:  # The user used .set
                mv_out = mvs[output_mv_idx]
                mv_out._values[:] = [self.algebra.simp_func(v) for v in mv_out._values]
                return None

        return MVType.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)

    def _call_binary(self, mv1, mv2):
        """ Specialization for binary operators. """
        keys_out, func, wrapped_func, _MVType = self[mv1, mv2]
        MVType = _MVType or MultiVector
        issymbolic = (mv1.issymbolic or mv2.issymbolic)
        if issymbolic:
            values_out = func(mv1.values(), mv2.values())
        else:
            values_out = wrapped_func(mv1.values(), mv2.values())

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MVType.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


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
        type_in = (type(mv), mv.keys())
        if type_in not in self.operator_dict:
            mv = self.make_symbolic_mvs((type_in,), (mv.shape,))[0]
            keys_out, func, MVType = do_codegen(self.codegen, mv, printer=self.printer, func_printer=self.func_printer, type_patterns=self.algebra.type_patterns[self.name])
            self.algebra.numspace[func.__name__] = wrapped_func = self.wrapper(func) if self.wrapper else func
            self.operator_dict[type_in] = OperatorDictOutput(keys_out, func, wrapped_func, MVType)
        return self.operator_dict[type_in]

    @resolve_and_expand
    def __call__(self, mv):
        mv = self._sanitize_mvs((mv,))[0]
        keys_out, func, wrapped_func, _MVType = self[mv]
        MVType = _MVType or MultiVector

        issymbolic = mv.issymbolic
        if issymbolic:
            values_out = func(mv.values())
        else:
            values_out = wrapped_func(mv.values())

        if issymbolic and self.algebra.simp_func:
            keys_out, values_out = self.filter(keys_out, values_out)

        return MVType.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)


class Registry(OperatorDict):
    def __getitem__(self, mvs: Tuple[MultiVector]):
        keys_in = tuple(mv.keys() for mv in mvs)
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            tapes = [TapeRecorder(algebra=self.algebra, expr=name, keys=keys)
                     for name, keys in zip(string.ascii_lowercase, keys_in)]
            keys_out, func = do_compile(self.codegen, *tapes)
            self.algebra.numspace[func.__name__] = wrapped_func = self.wrapper(func) if self.wrapper else func
            self.operator_dict[keys_in] = OperatorDictOutput(keys_out, func, wrapped_func)
        return self.operator_dict[keys_in]

    @resolve_and_expand
    def __call__(self, *mvs):
        if all(isinstance(mv, TapeRecorder) for mv in mvs):
            keys_out, func, wrapped_func, _MVType = self[mvs]
            MVType = _MVType or MultiVector
            expr = f"{func.__name__}({', '.join(mv.expr for mv in mvs)})"
            return TapeRecorder(self.algebra, keys=keys_out, expr=expr)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        values_in = tuple(mv.values() for mv in mvs)
        keys_out, func, wrapped_func, _MVType = self[mvs]
        MVType = _MVType or MultiVector
        values_out = wrapped_func(*values_in)

        return MVType.fromkeysvalues(self.algebra, keys=keys_out, values=values_out)
