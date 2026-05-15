from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Tuple, NamedTuple
from functools import wraps, cached_property
import inspect
import string

from sympy import Symbol, Expr, simplify
from sympy.printing.lambdarepr import LambdaPrinter

from kingdon.multivector import MultiVector, MultiVectorType, stack
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
        return res.asmvtype(MVType)
    return res


class CompiledExpression(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    :param wrapped_func: decorated func if a wrapper was provided, else identical to func.
    :param MVType: type of the output multivector. Defaults to :code:`MultiVector`.
    """
    algebra: "Algebra"
    keys_out: Tuple[int]
    func: Callable
    MVType: MultiVectorType = MultiVector
    output_mv_idx: int | None = None
    wrapped_func: Callable | None = None

    def __call__(self, *mvs):
        values_in = tuple(mv.values() for mv in mvs)
        values_out = self.func(*values_in)
        return self.MVType.fromkeysvalues(self.algebra, self.keys_out, values_out)

    def wrapped_call(self, *mvs):
        values_in = tuple(mv.values() for mv in mvs)
        values_out = self.wrapped_func(*values_in)
        return self.MVType.fromkeysvalues(self.algebra, self.keys_out, values_out)


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

        compiled_expr = self[mvs]
        issymbolic = any(mv.issymbolic for mv in mvs)
        if issymbolic:
            mv_out = compiled_expr(*mvs)
        else:
            mv_out = compiled_expr.wrapped_call(*mvs)

        if issymbolic and self.algebra.simp_func:
            mv_out = mv_out.filter(self.algebra.simp_func)
            if (output_mv_idx := compiled_expr.output_mv_idx):
                mvs[output_mv_idx].set(mv_out)
                return None
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
            mv_out = mv_out.filter(self.algebra.simp_func)

        return mv_out


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
            mv_out = mv_out.filter(self.algebra.simp_func)

        return mv_out


class Registry(OperatorDict):
    def __getitem__(self, mvs: Tuple[MultiVector]):
        keys_in = tuple(mv.keys() for mv in mvs)
        if keys_in not in self.operator_dict:
            # Make symbolic multivectors for each set of keys and generate the code.
            tapes = [TapeRecorder(algebra=self.algebra, expr=name, keys=keys)
                     for name, keys in zip(string.ascii_lowercase, keys_in)]
            keys_out, func = do_compile(self.codegen, *tapes)
            self.algebra.numspace[func.__name__] = wrapped_func = self.wrapper(func) if self.wrapper else func
            self.operator_dict[keys_in] = CompiledExpression(keys_out, func, self.algebra, wrapped_func)
        return self.operator_dict[keys_in]

    @resolve_and_expand
    def __call__(self, *mvs):
        if all(isinstance(mv, TapeRecorder) for mv in mvs):
            compiled_expr = self[mvs]
            expr = f"{compiled_expr.func.__name__}({', '.join(mv.expr for mv in mvs)})"
            return TapeRecorder(self.algebra, keys=compiled_expr.keys_out, expr=expr)

        # Make sure all inputs are multivectors. If an input is not, assume its scalar.
        mvs = [mv if isinstance(mv, MultiVector) else MultiVector.fromkeysvalues(self.algebra, (0,), (mv,))
               for mv in mvs]
        if any((mvs[0].algebra != mv.algebra) for mv in mvs[1:]):
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        compiled_expr = self[mvs]
        return compiled_expr.wrapped_call(*mvs)
