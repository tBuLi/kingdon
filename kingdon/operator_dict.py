from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Callable
from typing import NamedTuple
from functools import wraps, cached_property
import inspect
import string
from types import GenericAlias

import sympy

from kingdon.multivector import MultiVector, MultiVectorType, Scalar, stack
from kingdon.codegen import do_compile_symbolic, do_compile
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


def do_operation(*mvs, codegen, algebra, MVType=None) -> MultiVector:
    """
    This function just does the operation directly on the MV's, no codegen is performed.
    This is used for large algebras, where codegen is too costly.
    The result is the multivector resulting from :code:`codegen(*mvs)`.
    """
    MVType = MVType or algebra.mvtype
    mvs = [mv if isinstance(mv, MultiVector) else algebra.mvtype.fromkeysvalues(algebra, (0,), [mv,])
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

    Here, :code:`ops.gp` is a function that takes two symbolic :class:`~kingdon.multivector.MultiVector` as input
    and outputs a single :class:`~kingdon.multivector.MultiVector`.
    """
    name: str
    codegen: Callable
    algebra: "Algebra"
    operator_dict: dict = field(default_factory=dict, init=False)
    codegen_symbolcls: Callable = field(default=None, repr=False)
    lambdifier: Callable = field(default=None, repr=False)
    lambdifier_kwargs: dict = field(default_factory=dict, repr=False)
    wrapper: Callable = field(default=None, repr=False)
    values_asarray: Callable = field(default=None, repr=False)

    def __post_init__(self):
         # If the user forces a different codegen settings for this operator then give them what they want.
        if not self.codegen_symbolcls:
            self.codegen_symbolcls = self.algebra.codegen_symbolcls
        if not self.lambdifier and self.algebra.lambdifier:
            self.lambdifier = self.algebra.lambdifier
        if not self.wrapper and self.algebra.wrapper:
            self.wrapper = self.algebra.wrapper
        if not self.values_asarray:
            self.values_asarray = self.algebra.values_asarray

    def __len__(self):
        return len(self.operator_dict)

    def _make_symbolic_mv(self, name, keys, shape, mvtype, mvtypehint) -> MultiVector:
        depth = mvtypehint[1] if isinstance(mvtypehint, tuple) else 0  # Must come from the type-hint.
        if depth is None: depth = shape[0]  # If the type hint was MultiVector[None] than the depth should be taken from the input
        # During codegen we need to ignore full_layout
        if not depth:
            return mvtype.fromname(self.algebra, name, keys, symbolcls=self.codegen_symbolcls, full_layout=False)
        return stack([mvtype.fromname(self.algebra, f'{name}_{k}', keys, symbolcls=self.codegen_symbolcls, full_layout=False)
                      for k in range(depth)])

    def make_symbolic_mvs(self, types_in: tuple[tuple[type, tuple[int]]], shapes_in: tuple[tuple[int]]) -> tuple[MultiVector]:
        return tuple(
            self._make_symbolic_mv(name, keys, shape, MVtype, MVTypeHint)
            for (name, MVTypeHint), (MVtype, keys), shape in zip(self.codegen_input_types.items(), types_in, shapes_in)
        )

    def __getitem__(self, mvs: tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if types_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            symbolic_mvs = self.make_symbolic_mvs(types_in, shapes_in)
            compiled = self.operator_dict[types_in] = self.algebra.compile(self.codegen, *symbolic_mvs, lambdifier=self.lambdifier, wrapper=self.wrapper, values_asarray=self.values_asarray, **self.lambdifier_kwargs)
            self.algebra.numspace[compiled.func.__name__] = compiled.wrapped_func
        return self.operator_dict[types_in]

    def __contains__(self, mvs: tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        return types_in in self.operator_dict

    def __iter__(self):
        return iter(self.operator_dict)

    @cached_property
    def codegen_input_types(self):
        return {name: self.algebra.mvtype if p.annotation in (inspect.Parameter.empty, "MultiVector") else p.annotation
                for name, p in inspect.signature(self.codegen).parameters.items()}

    @cached_property
    def codegen_output_type(self):
        return_annotation = inspect.signature(self.codegen).return_annotation
        return self.algebra.mvtype if return_annotation in (inspect.Parameter.empty, "MultiVector") else return_annotation

    def _simplify(self, mv: MultiVector) -> MultiVector:
        """
        Apply the algebra's :code:`simp_func` to the coefficients of `mv`. Coefficients that simplify
        to zero are dropped, unless we are in full_layout mode.
        """
        if self.algebra.full_layout:
            return mv.map(self.algebra.simp_func)
        return mv.filter(self.algebra.simp_func, map=True)

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
        mv_out = compiled_expr(*mvs)

        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mvs[output_mv_idx].set(self._simplify(mvs[output_mv_idx]))
            else:
                mv_out = self._simplify(mv_out)
        return mv_out

    def _call_binary(self, mv1, mv2):
        """ Specialization for binary operators. """
        compiled_expr = self[mv1, mv2]
        mv_out = compiled_expr(mv1, mv2)

        issymbolic = (mv1.issymbolic or mv2.issymbolic)
        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mvs = [mv1, mv2]
                mvs[output_mv_idx].set(self._simplify(mvs[output_mv_idx]))
            else:
                mv_out = self._simplify(mv_out)

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
            symbolic_mv = self.make_symbolic_mvs((type_in,), (mv.shape,))[0]
            compiled = self.operator_dict[type_in] = self.algebra.compile(self.codegen, symbolic_mv, lambdifier=self.lambdifier, wrapper=self.wrapper, values_asarray=self.values_asarray, **self.lambdifier_kwargs)
            self.algebra.numspace[compiled.func.__name__] = compiled.wrapped_func
        return self.operator_dict[type_in]

    @resolve_and_expand
    def __call__(self, mv):
        mv = self._sanitize_mvs((mv,))[0]
        compiled_expr = self[mv]
        mv_out = compiled_expr(mv)

        issymbolic = mv.issymbolic
        if issymbolic and self.algebra.simp_func:
            if (output_mv_idx := compiled_expr.output_mv_idx) is not None:  # A function that contains .set
                mv[output_mv_idx].set(self._simplify(mv))
            else:
                mv_out = self._simplify(mv_out)

        return mv_out


class Registry(OperatorDict):
    def __getitem__(self, mvs: tuple[MultiVector]):
        types_in = tuple((type(mv), mv.keys()) for mv in mvs)
        shapes_in = tuple(mv.shape for mv in mvs)
        if types_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            # tapes = [TapeRecorder(algebra=self.algebra, expr=name, mvtype=MVType, keys=keys)
            #          for name, (MVType, keys) in zip(string.ascii_lowercase, types_in)]
            tapes = self.make_symbolic_mvs(types_in, shapes_in)
            compiled = self.operator_dict[types_in] = self.algebra.compile(self.codegen, *tapes, symbolic=False, wrapper=self.wrapper, values_asarray=self.values_asarray)
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
        mvs = [mv if isinstance(mv, MultiVector) else self.algebra.mvtype.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        compiled_expr = self[mvs]
        return compiled_expr(*mvs)

    def _make_symbolic_mv(self, name, keys, shape, mvtype, mvtypehint) -> TapeRecorder:
        depth = mvtypehint[1] if isinstance(mvtypehint, tuple) else 0  # Must come from the type-hint.
        if depth is None: depth = shape[1]  # If the type hint was MultiVector[None] than the depth should be taken from the input
        if not depth:
            return TapeRecorder.fromname(self.algebra, name, keys, mvtype=mvtype)
        return TapeRecorder.fromname(self.algebra, name, keys, mvtype=mvtype, shape=(depth,))

    def make_symbolic_mvs(self, types_in: tuple[tuple[type, tuple[int]]], shapes_in: tuple[tuple[int]]) -> tuple[TapeRecorder]:
        return tuple(
            self._make_symbolic_mv(name, keys, shape, MVType, MVTypeHint)
            for (name, MVTypeHint), (MVType, keys), shape in zip(self.codegen_input_types.items(), types_in, shapes_in)
        )
