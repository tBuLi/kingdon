from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple, NamedTuple, Type, List
from functools import wraps, cached_property
import inspect
import string
from types import GenericAlias

from sympy import Symbol, Expr, simplify
from sympy.printing.lambdarepr import LambdaPrinter

from kingdon.multivector import MultiVector, MultiVectorType, Scalar, stack
from kingdon.codegen import do_compile_symbolic, do_compile, KingdonPrinter
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
        return res.asmvtype(MVType)
    return res


@dataclass
class OperatorDict(Mapping):
    """
    A dict-like object which performs codegen of a particular operator,
    and caches the result for future use. For example, to generate the geometric product,
    we create an OperatorDict as follows::

        import kingdon.operators as ops
        alg = Algebra(3, 0, 1)
        gp = OperatorDict('gp', codegen=ops.gp, algebra=alg)

    Here, :code:`ops.gp` is a function that takes two symbolic :class:`MultiVector` as input
    and outputs a single :class:`MultiVector`.
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

    def _make_symbolic_mv(self, name, keys, shape, mvtype, mvtypehint) -> MultiVector:
        depth = mvtypehint[1] if isinstance(mvtypehint, tuple) else 0  # Must come from the type-hint.
        if depth is None: depth = shape[1]  # If the type hint was MultiVector[None] than the depth should be taken from the input
        if not depth:
            return mvtype.fromname(self.algebra, name, keys, symbolcls=self.codegen_symbolcls)
        return stack([mvtype.fromname(self.algebra, f'{name}_{k}', keys, symbolcls=self.codegen_symbolcls)
                      for k in range(depth)])

    def make_symbolic_mvs(self, types_in: Tuple[Tuple[Type, Tuple[int]]], shapes_in: Tuple[Tuple[int]]) -> tuple[MultiVector]:
        return tuple(
            self._make_symbolic_mv(name, keys, shape, MVtype, MVTypeHint)
            for (name, MVTypeHint), (MVtype, keys), shape in zip(self.codegen_input_types.items(), types_in, shapes_in)
        )

    def __getitem__(self, mvs: Tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if types_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            archetypes = self.make_symbolic_mvs(types_in, shapes_in)
            compiled = self.operator_dict[types_in] = self.algebra.compile(self.codegen, *archetypes, printer=self.printer, func_printer=self.func_printer, wrapper=self.wrapper)
            self.algebra.numspace[compiled.func.__name__] = compiled.wrapped_func
        return self.operator_dict[types_in]

    def __contains__(self, mvs: Tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        return types_in in self.operator_dict

    def __iter__(self):
        return iter(self.operator_dict)

    @cached_property
    def codegen_input_types(self):
        return {name: MultiVector if p.annotation in (inspect.Parameter.empty, "MultiVector") else p.annotation
                for name, p in inspect.signature(self.codegen).parameters.items()}

    @cached_property
    def codegen_output_type(self):
        return_annotation = inspect.signature(self.codegen).return_annotation
        return MultiVector if return_annotation in (inspect.Parameter.empty, "MultiVector") else return_annotation

    def _sanitize_mvs(self, mvs: tuple[MultiVector]):
        """
        Make sure all inputs match their type annotations.
        If an input is expected to be a MultiVector but is not, cast it to a scalar.
        # TODO: make sure that if the typehint was trivial, then we do not want to pay the price
        #   of fixing an interesting typehint.
        """
        if len(mvs) == 1:
            mv = mvs[0]
            mv = mv if isinstance(mv, MultiVector) else Scalar.fromkeysvalues(self.algebra, (0,), [mv])
            return (mv,)
        if len(mvs) == 2:
            mv1, mv2 = mvs
            mv1 = mv1 if isinstance(mv1, MultiVector) else Scalar.fromkeysvalues(self.algebra, (0,), [mv1])
            mv2 = mv2 if isinstance(mv2, MultiVector) else Scalar.fromkeysvalues(self.algebra, (0,), [mv2])
            mvs = (mv1, mv2)
        else:
            mvs = [mv if isinstance(mv, MultiVector) else Scalar.fromkeysvalues(self.algebra, (0,), [mv])
                   for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")
        return mvs

    @resolve_and_expand
    def __call__(self, *mvs):
        mvs = self._sanitize_mvs(mvs)
        if len(mvs) == 2:
            return self._call_binary(*mvs)

        compiled_expr = self[mvs]
        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic:
            mv_out = compiled_expr(*mvs)
        else:
            mv_out = compiled_expr.wrapped_call(*mvs)

        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mvs[output_mv_idx].set(mvs[output_mv_idx].filter(self.algebra.simp_func, map=True))
            else:
                mv_out = mv_out.filter(self.algebra.simp_func, map=True)
        return mv_out

    def _call_binary(self, mv1, mv2):
        """ Specialization for binary operators. """
        compiled_expr = self[mv1, mv2]
        issymbolic = (mv1.issymbolic or mv2.issymbolic)
        if issymbolic:
            mv_out = compiled_expr(mv1, mv2)
        else:
            mv_out = compiled_expr.wrapped_call(mv1, mv2)

        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mvs = [mv1, mv2]
                mvs[output_mv_idx].set(mvs[output_mv_idx].filter(self.algebra.simp_func, map=True))
            else:
                mv_out = mv_out.filter(self.algebra.simp_func, map=True)

        return mv_out


class UnaryOperatorDict(OperatorDict):
    """
    Specialization of OperatorDict for unary operators. In the
    case of unary operators, we can do away with all of the overhead that is necessary for
    operators that act on multiple multivectors.
    """
    def __contains__(self, mv: MultiVector):
        types_in = (type(mv), mv.keys())
        return types_in in self.operator_dict

    def __getitem__(self, mv: MultiVector):
        type_in = (type(mv), mv.keys())
        if type_in not in self.operator_dict:
            archetype = self.make_symbolic_mvs((type_in,), (mv.shape,))[0]
            compiled = self.operator_dict[type_in] = self.algebra.compile(self.codegen, archetype, printer=self.printer, func_printer=self.func_printer, wrapper=self.wrapper)
            self.algebra.numspace[compiled.func.__name__] = compiled.wrapped_func
        return self.operator_dict[type_in]

    @resolve_and_expand
    def __call__(self, mv):
        mv = self._sanitize_mvs((mv,))[0]
        compiled_expr = self[mv]

        issymbolic = mv.issymbolic
        if issymbolic:
            mv_out = compiled_expr(mv)
        else:
            mv_out = compiled_expr.wrapped_call(mv)

        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mv[output_mv_idx].set(mv.filter(self.algebra.simp_func, map=True))
            else:
                mv_out = mv_out.filter(self.algebra.simp_func, map=True)

        return mv_out


class Registry(OperatorDict):
    def __getitem__(self, mvs: Tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if types_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            # tapes = [TapeRecorder(algebra=self.algebra, expr=name, mvtype=MVType, keys=keys)
            #          for name, (MVType, keys) in zip(string.ascii_lowercase, types_in)]
            tapes = self.make_symbolic_mvs(types_in, shapes_in)
            compiled = self.operator_dict[types_in] = self.algebra.compile(self.codegen, *tapes, symbolic=False)
            self.algebra.numspace[compiled.func.__name__] = compiled.wrapped_func
        return self.operator_dict[types_in]

    @resolve_and_expand
    def __call__(self, *mvs):
        if all(isinstance(mv, TapeRecorder) for mv in mvs):
            types_in = tuple((mv.mvtype, mv.keys()) for mv in mvs)
            compiled_expr = self.operator_dict[types_in]
            expr = f"{compiled_expr.func.__name__}({', '.join(mv.expr for mv in mvs)})"
            return TapeRecorder(self.algebra, mvtype=compiled_expr.mvtype, keys=compiled_expr.keys_out, expr=expr)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        compiled_expr = self[mvs]
        return compiled_expr.wrapped_call(*mvs)

    def _make_symbolic_mv(self, name, keys, shape, mvtype, mvtypehint) -> TapeRecorder:
        depth = mvtypehint[1] if isinstance(mvtypehint, tuple) else 0  # Must come from the type-hint.
        if depth is None: depth = shape[1]  # If the type hint was MultiVector[None] than the depth should be taken from the input
        if not depth:
            return TapeRecorder.fromname(self.algebra, name, keys, mvtype=mvtype)
        return TapeRecorder.fromname(self.algebra, name, keys, mvtype=mvtype, shape=(depth,))

    def make_symbolic_mvs(self, types_in: Tuple[Tuple[Type, Tuple[int]]], shapes_in: Tuple[Tuple[int]]) -> tuple[TapeRecorder]:
        return tuple(
            self._make_symbolic_mv(name, keys, shape, MVType, MVTypeHint)
            for (name, MVTypeHint), (MVType, keys), shape in zip(self.codegen_input_types.items(), types_in, shapes_in)
        )
