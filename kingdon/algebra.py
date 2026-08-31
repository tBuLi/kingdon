import operator
import re
from itertools import product
from functools import partial, reduce
from collections import Counter
from dataclasses import dataclass, field, fields, InitVar
from collections.abc import Mapping, Callable
from typing import List, Tuple
import warnings

try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        return property(lru_cache()(func))

import sympy

from kingdon.codegen import (
    do_compile_symbolic,
    do_compile,
    KingdonPrinter,
)
import kingdon.operators as ops
from kingdon.operator_dict import OperatorDict, UnaryOperatorDict, Registry, do_operation, resolve_and_expand
from kingdon.polynomial import RationalPolynomial
from kingdon.matrixreps import matrix_rep
from kingdon.multivector import (
    MultiVector, MultiVectorType,
    KVector, Scalar, Vector, Bivector, Trivector, Quadvector, Pentavector, Hexavector, Heptavector, Octovector, # k-vectors
    Bireflection, # compositions
    Direction, EVector, UPoint, Point, Translation,  # PGA Types.
)
from kingdon.graph import GraphWidget
from kingdon.codegen import resolve_layout, CompiledExpression

operation_field = partial(field, default_factory=dict, init=False, repr=False, compare=False)


@dataclass
class Algebra:
    """
    A Geometric (Clifford) algebra with :code:`p` positive dimensions,
    :code:`q` negative dimensions, and :code:`r` null dimensions.

    The default settings of :code:`cse = simplify = True` usually strike a good balance between
    initiation times and subsequent code execution times.

    :param p:  number of positive dimensions.
    :param q:  number of negative dimensions.
    :param r:  number of null dimensions.
    :param signature: Optional signature of the algebra, e.g. [0, 1, 1] for 2DPGA.
        Mutually exclusive with `p`, `q`, `r`.
    :param start_index: Optionally set the start index of the dimensions. For PGA this defaults to `0`, otherwise `1`.
    :param basis: Custom basis order, e.g. `["e", "e1", "e2", "e0", "e20", "e01", "e12", "e012"]` for 2DPGA.
    :param cse: If :code:`True` (default), attempt Common Subexpression Elimination (CSE)
        on symbolically optimized expressions.
    :param graded: If :code:`True` (default is :code:`False`), perform binary and unary operations on a graded basis.
        This will still be more sparse than computing with a full multivector, but not as sparse as possible.
        It does however, vastly reduce the number of possible expressions that have to be symbolically optimized.
    :param extra_types: multivector types to use in addition to the standard ones already provided by kingdon. See :ref:`multivector types <Multivector Types>`.
    :param types: Complete list of multivector types to use. This will replace the standard ones. See :ref:`multivector types <Multivector Types>`.
    :param simplify: If :code:`True` (default), we attempt to simplify as much as possible. Setting this to
        :code:`False` will reduce the number of calls to simplify. However, it seems that :code:`True` is still faster,
        because it keeps sympy expressions from growing too large, which makes both symbolic computations and
        printing into a python function slower.
    :param wrapper: A function that is always applied to the generated functions as a decorator. For example,
        using :code:`numba.njit` as a wrapper will ensure that all kingdon code is jitted using numba.
    :param values_asarray: An array construction function that is always applied to values of a multivector upon
        creation of a new mv. For example, this could be :code:`numpy.array` or :code:`torch.asarray` to ensure
        that mv's are always your favorite arrays.
    :param symbolcls: The symbol class used for symbolic multivectors. By default, this :class:`sympy.Symbol`.
    :param codegen_symbolcls: The symbol class used during codegen. By default, this is our own fast
        :class:`~kingdon.polynomial.RationalPolynomial` class.
    :param printer: Sympy code printer used for codegen, see `https://docs.sympy.org/latest/modules/printing.html`.
    :param simp_func: This function is applied as a filter function to every multivector coefficient.
    :param pretty_blade: character to use for basis blades when pretty printing to string. Default is 𝐞.
    :param large: if true this is considered a large algebra. This means various cashing options are removed to save
        memory, and codegen is replaced by direct computation since codegen is very resource intensive for big
        expressions. By default, algebras of :math:`d > 6` are considered large, but the user can override this setting
        because also in large algebras it is still true that the generated code will perform order(s) of magnitude
        better than direct computation.
    """
    p: int = field(default=0, repr=False, compare=False)
    q: int = field(default=0, repr=False, compare=False)
    r: int = field(default=0, repr=False, compare=False)
    d: int = field(init=False, repr=False, compare=False)  # Total number of dimensions
    signature: List[int] = field(default=None)
    start_index: int = field(default=None, repr=False, compare=False)
    basis: List[str] = field(default_factory=list)

    # Clever dictionaries that cache previously symbolically optimized lambda functions between elements.
    gp: OperatorDict = operation_field(metadata={'codegen': ops.gp,})  # geometric product
    sw: OperatorDict = operation_field(metadata={'codegen': ops.sw})  # conjugation
    cp: OperatorDict = operation_field(metadata={'codegen': ops.cp,})  # commutator product
    acp: OperatorDict = operation_field(metadata={'codegen': ops.acp,})  # anti-commutator product
    ip: OperatorDict = operation_field(metadata={'codegen': ops.ip,})  # inner product
    sp: OperatorDict = operation_field(metadata={'codegen': ops.sp,})  # Scalar product
    lc: OperatorDict = operation_field(metadata={'codegen': ops.lc,})  # left-contraction
    rc: OperatorDict = operation_field(metadata={'codegen': ops.rc,})  # right-contraction
    op: OperatorDict = operation_field(metadata={'codegen': ops.op,})  # exterior product
    rp: OperatorDict = operation_field(metadata={'codegen': ops.rp,})  # regressive product
    proj: OperatorDict = operation_field(metadata={'codegen': ops.proj})  # projection
    add: OperatorDict = operation_field(metadata={'codegen': ops.add,})  # add
    sub: OperatorDict = operation_field(metadata={'codegen': ops.sub,})  # sub
    div: OperatorDict = operation_field(metadata={'codegen': ops.div})  # division
    # Unary operators
    inv: UnaryOperatorDict = operation_field(metadata={'codegen': ops.inv})  # inverse
    neg: UnaryOperatorDict = operation_field(metadata={'codegen': ops.neg,})  # negate
    reverse: UnaryOperatorDict = operation_field(metadata={'codegen': ops.reverse,})  # reverse
    involute: UnaryOperatorDict = operation_field(metadata={'codegen': ops.involute,})  # grade involution
    conjugate: UnaryOperatorDict = operation_field(metadata={'codegen': ops.conjugate,})  # clifford conjugate
    sqrt: UnaryOperatorDict = operation_field(metadata={'codegen': ops.sqrt})  # Square root
    polarity: UnaryOperatorDict = operation_field(metadata={'codegen': ops.polarity})
    unpolarity: UnaryOperatorDict = operation_field(metadata={'codegen': ops.unpolarity})
    hodge: UnaryOperatorDict = operation_field(metadata={'codegen': ops.hodge,})
    unhodge: UnaryOperatorDict = operation_field(metadata={'codegen': ops.unhodge,})
    normsq: UnaryOperatorDict = operation_field(metadata={'codegen': ops.normsq})  # norm squared
    outerexp: UnaryOperatorDict = operation_field(metadata={'codegen': ops.outerexp})
    outersin: UnaryOperatorDict = operation_field(metadata={'codegen': ops.outersin})
    outercos: UnaryOperatorDict = operation_field(metadata={'codegen': ops.outercos})
    outertan: UnaryOperatorDict = operation_field(metadata={'codegen': ops.outertan})
    registry: dict = field(default_factory=dict, repr=False, compare=False)  # Dict of all operator dicts. Should be extended using Algebra.jit
    numspace: dict = field(default_factory=dict, repr=False, compare=False)  # Namespace for numerical functions

    # Mappings from binary to canonical reps. e.g. 0b01 = 1 <-> 'e1', 0b11 = 3 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False, compare=False)
    bin2canon: dict = field(init=False, repr=False, compare=False)

    # Options for the algebra
    cse: bool = field(default=True, repr=False, compare=False)  # Common Subexpression Elimination (CSE)
    graded: bool = field(default=False, repr=False)  # If true, precompute products per grade.
    pretty_blade: str = field(default='𝐞', repr=False, compare=False)
    pretty_digits: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # TODO: this can be defined outside Algebra
    large: bool = field(default=None, repr=False, compare=False)
    extra_types: InitVar[list | None] = None
    types: list = field(default_factory=list, repr=False, compare=False)
    _type_layouts: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # Codegen & call customization.
    symbolcls: object = field(default=None, repr=False, compare=False)
    # The symbol class used in codegen. By default, use our own fast RationalPolynomial class.
    codegen_symbolcls: object = field(default=RationalPolynomial.fromname, repr=False, compare=False)
    # The sympy style printer and evaluator printer used to generate the code with sympy-style printing.
    printer: sympy.printing.lambdarepr.LambdaPrinter = field(default=None, repr=False, compare=False)
    func_printer: KingdonPrinter = field(default=None, repr=False, compare=False)
    # Wrapper function applied to the codegen generated functions.
    wrapper: Callable = field(default=None, repr=False, compare=False)
    # Constructor to be called on the values upon mv creation.
    values_asarray: Callable = field(default=list, repr=False, compare=False)

    # This simplify func is applied to every component after a symbolic expression is called, to simplify and filter by.
    simp_func: Callable = field(default=lambda v: v if not isinstance(v, sympy.Expr) else sympy.simplify(sympy.expand(v)), repr=False, compare=False)

    signs: dict = field(init=False, repr=False, compare=False)
    blades: "BladeDict" = field(init=False, repr=False, compare=False)
    pss: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, extra_types):
        if self.signature is not None:
            counts = Counter(self.signature)
            self.p, self.q, self.r = counts[1], counts[-1], counts[0]
            if self.p + self.q + self.r != len(self.signature):
                raise TypeError('Unsupported signature.')
        else:
            if self.r == 1:  # PGA, so put r first.
                self.signature = [0] * self.r + [1] * self.p + [-1] * self.q
            else:
                self.signature = [1] * self.p + [-1] * self.q + [0] * self.r

        if self.start_index is None:
            self.start_index = 0 if self.r == 1 else 1

        self.d = self.p + self.q + self.r

        if self.d + self.start_index <= 10:
            self.pretty_digits = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',}
        else:
            # Use superscript above 10 because that is almost the entire alphabet.
            self.pretty_digits = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                'A': 'ᴬ', 'B': 'ᴮ', 'C': 'ᶜ', 'D': 'ᴰ', 'E': 'ᴱ',
                'F': 'ᶠ', 'G': 'ᴳ', 'H': 'ᴴ', 'I': 'ᴵ', 'J': 'ᴶ',
                'K': 'ᴷ', 'L': 'ᴸ', 'M': 'ᴹ', 'N': 'ᴺ', 'O': 'ᴼ',
                'P': 'ᴾ', 'R': 'ᴿ', 'Q': 'Q', 'S': 'ˢ', 'T': 'ᵀ', 'U': 'ᵁ',
                'V': 'ⱽ', 'W': 'ᵂ', 'X': 'ˣ', 'Y': 'ʸ', 'Z': 'ᶻ'
            }

        # Setup mapping from binary to canonical string rep and vise versa
        if self.basis:
            assert len(self.basis) == len(self)
            assert self.basis == sorted(self.basis, key=len)  # The basis has to be ordered by grade.
            assert all(eJ[0] == 'e' for eJ in self.basis)
            vecs = [eJ[1:] for eJ in self.basis if len(eJ) == 2]
            self.start_index = int(min(vecs))
            vec2bin = {vec: 2 ** j for j, vec in enumerate(vecs)}
            self.canon2bin = {eJ: reduce(operator.xor, (vec2bin[v] for v in eJ[1:]), 0)
                              for eJ in self.basis}
            self.bin2canon = {J: eJ for eJ, J in sorted(self.canon2bin.items(), key=lambda x: x[1])}
        else:
            digits = list(self.pretty_digits)
            self.bin2canon = {
                eJ: 'e' + ''.join(digits[ei + self.start_index] for ei in range(0, self.d) if eJ & 2**ei)
                for eJ in range(2 ** self.d)
            }
            self.canon2bin = dict(sorted({c: b for b, c in self.bin2canon.items()}.items(), key=lambda x: (len(x[0]), x[0])))
            self.basis = list(self.canon2bin)

        self.signs = DefaultKeyDict(self._compute_sign)

        if self.large is None:
            self.large = self.d > 6

        if self.large:  # Do direct computation instead of codegen
            self.registry = {f.name: self.wrapper(resolve_and_expand(partial(do_operation, codegen=codegen, algebra=self)))
                                     if self.wrapper else resolve_and_expand(partial(do_operation, codegen=codegen, algebra=self))
                             for f in fields(self) if (codegen := f.metadata.get('codegen'))}
        else:
            # Prepare OperatorDict's
            self.registry = {f.name: f.type(name=f.name, algebra=self, **f.metadata)
                             for f in fields(self) if 'codegen' in f.metadata}
        for name, op in self.registry.items():
            setattr(self, name, op)

        self._kvectors = []
        if not self.types:
            kvectors = [Scalar, Vector, Bivector, Trivector, Quadvector, Pentavector, Hexavector, Heptavector, Octovector]
            self._kvectors = kvectors[:self.d+1]
            self.types = [*self._kvectors]
            if self.d >= 2: self.types.extend([Bireflection])
            if extra_types: self.types.extend(extra_types)
        # Dynamically generate classes for types if they are not already.
        self.types = [type(t['name'], (MultiVector,), {'layout': t['layout']}) if isinstance(t, dict) else t
                      for t in self.types]
        self._type_layouts = {cls: layout for cls in self.types
                              if (layout := self._bind_layout(cls, name='x'))}  # an empty layout matches an empty result at zero cost in resolve_layout, and would beat every other type.

        # Add mv constructors to the algebra
        for cls in self._type_layouts: setattr(self, cls.__name__.lower(), partial(cls, self))
        for k, cls in enumerate(self._kvectors):
            if self.d - k < len(self._kvectors):
                setattr(self, f"pseudo{cls.__name__.lower()}", partial(self._kvectors[self.d - k], self))

        # Blades are not precomputed for large algebras, except for basis vectors.
        self.blades = BladeDict(algebra=self, lazy=self.large)
        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

    @classmethod
    def fromname(cls, name: str, extra_types=None, **kwargs):
        """
        Initialize a well known algebra by its name. Options are 2DPGA, 3DPGA, and STAP.
        This uses sensible ordering of the basis vectors in the basis blades to avoid minus superfluous signs.
        """
        extra_pga_types = [Direction, EVector, UPoint, Point, Translation]
        if extra_types: extra_pga_types.extend(extra_types)
        if name == '2DPGA':
            basis = ["e", "e1", "e2", "e0", "e20", "e01", "e12", "e012"]
            return cls(2, 0, 1, basis=basis, extra_types=extra_pga_types, **kwargs)
        elif name == '3DPGA':
            basis = ["e", "e1", "e2", "e3", "e0",
                     "e01", "e02", "e03", "e12", "e31", "e23",
                     "e032", "e013", "e021", "e123", "e0123"]
            return cls(3, 0, 1, basis=basis, extra_types=extra_pga_types, **kwargs)
        elif name == 'STAP':
            basis = ["e", "e0", "e1", "e2", "e3", "e4",
                     "e01", "e02", "e03", "e40", "e12", "e31", "e23", "e41", "e42", "e43",
                     "e234", "e314", "e124", "e123", "e014", "e024", "e034", "e032", "e013", "e021",
                     "e0324", "e0134", "e0214", "e0123", "e1234", "e01234"]
            return cls(3, 1, 1, basis=basis, extra_types=extra_pga_types, **kwargs)
        else:
            raise ValueError("No algebra by this name is known.")

    def __len__(self):
        return 2 ** self.d

    def indices_for_grade(self, grade: int):
        """
        Function that returns a generator for all the indices for a given grade. E.g. in 2D VGA, this returns

        .. code-block ::

            >>> alg = Algebra(2)
            >>> tuple(alg.indices_for_grade(1))
            (1, 2)

        The indices are returned in the same order as the basis blades.
        """
        return (k for k in self.canon2bin.values() if ops._bit_count(k) == grade)

    def indices_for_grades(self, grades: Tuple[int]):
        """
        Function that returns a generator for all the indices from a sequence of grades.
        E.g. in 2D VGA, this returns

        .. code-block ::

            >>> alg = Algebra(2)
            >>> tuple(alg.indices_for_grades((1, 2)))
            (1, 2, 3)

        The indices are returned in the same order as the basis blades.
        """
        grades = tuple(sorted(grades))
        return (k for k in self.canon2bin.values() if ops._bit_count(k) in grades)

    @cached_property
    def matrix_basis(self):
        return matrix_rep(self.p, self.q, self.r, signature=self.signature)

    @cached_property
    def frame(self) -> list:
        r"""
        The set of orthogonal basis vectors, :math:`\{ e_i \}`. Note that for a frame linear independence suffices,
        but we already have orthogonal basis vectors so why not use those?
        """
        return [self.blades[self.bin2canon[2**j]] for j in range(0, self.d)]

    @cached_property
    def reciprocal_frame(self) -> list:
        r"""
        The reciprocal frame is a set of vectors :math:`\{ e^i \}` that satisfies
        :math:`e^i \cdot e_j = \delta^i_j` with the frame vectors :math:`\{ e_i \}`.
        """
        return [v.inv() for v in self.frame]

    def _compute_sign(self, bin_pair: tuple[int, int]):
        """ Computes the sign between two basis blades in binary rep. """
        I, J = bin_pair
        eI, eJ = self.bin2canon[I], self.bin2canon[J]
        # Compute the number of swaps of orthogonal vectors needed to order the basis vectors.
        swaps, prod, eliminated = _swap_blades(eI[1:], eJ[1:], self.bin2canon[I ^ J][1:])

        # Remove even powers of basis-vectors.
        sign = -1 if swaps % 2 else 1
        for key in eliminated:
            sign *= self.signature[int(key, base=len(self.pretty_digits)) - self.start_index]
        return sign

    @cached_property
    def cayley(self):
        """ Cayley table of the algebra. """
        cayley = {}
        for (eI, I), (eJ, J) in product(self.canon2bin.items(), repeat=2):
            if sign := self.signs[I, J]:
                sign = '-' if sign == -1 else ''
                cayley[eI, eJ] = f'{sign}{self.bin2canon[I ^ J]}'
            else:
                cayley[eI, eJ] = f'0'
        return cayley

    def register(self, expr=None, /, *, name=None, symbolic=False):
        """
        Register a function with the algebra to optimize its execution times. Deprecated in favor of :meth:`~kingdon.algebra.Algebra.jit`.
        """
        warnings.warn("Use @alg.jit instead of @alg.register", FutureWarning)
        return self.jit(expr, name=name, symbolic=symbolic)

    def jit(self, expr=None, /, *, name=None, symbolic=False, codegen_symbolcls=None, printer=None, func_printer=None, wrapper=None, values_asarray=None):
        """
        Mark a function for Just-in-time (JIT) compilation to optimize its execution times.
        The function must accept multivectors as input arguments and is assumed to either
        return a single multivector or to :code:`set` one of the input multivectors.

        Examples:

        .. code-block ::

            @alg.jit(symbolic=True)
            def proj(a, b):
                return (a | b) / b

            @alg.jit(symbolic=True)
            def proj_allocated(a, b, c):
                c.set((a | b) / b)

            @alg.jit(symbolic=True)
            def proj(mvs: MultiVector[2]):
                a, b = mvs
                return (a | b) / b

        The examples above show three different ways to compile a function.
        Firstly and most straightforwardly, simply decorate a function that takes MultiVectors as input and a single MultiVector as output.
        Secondly, rather than returning a MultiVector, one can modify one existing MultiVector in place using the `set` method.
        This is useful in combination with e.g. pre-allocated memory like is sometimes the case with PyTorch tensors.
        Thirdly, using the type annotation `MultiVector[N]` we can signal to compile that the function takes a multivector of shape (:, N).

        With default settings (symbolic=False), the decorator will ensure that every GA unary or binary
        operator is replaced by the corresponding numerical function, and produces
        numerically much more performant code.
        However, when `symbolic=True` the expression is symbolically optimized before being turned
        into a numerical function.
        This typically results in even more performant code, at the expense of extra cost for the first execution.

        :param expr: Python function of a valid kingdon GA expression.
        :param name: (optional) name by which the function will be known to the algebra.
            By default, this is the `expr.__name__`.
        :param symbolic: (optional) If true, the expression is symbolically optimized.
            By default this is False, given the cost of optimizing large expressions.
        :param codegen_symbolcls: (optional) The class to use for symbolic multivectors.
            By default the codegen_symbolcls from Algebra is used.
        :param printer: (optional) The sympy style printer used to generate the code with sympy-style printing.
            By default the printer from Algebra is used.
        :param func_printer: (optional) The sympy style evaluator printer used to generate the code with sympy-style printing.
            By default the func_printer from Algebra is used.
        :param wrapper: (optional) The wrapper function used to wrap the compiled function.
            By default the wrapper from Algebra is used.
        :param values_asarray: (optional) The values_asarray function used to cast the values to the correct array type.
            By default values_asarray from Algebra is used.
        """
        def wrap(expr, name=None, symbolic=False):
            if name is None:
                name = expr.__name__

            if not symbolic:
                self.registry[name] = Registry(name, codegen=expr, algebra=self, wrapper=wrapper, values_asarray=values_asarray)
            else:
                self.registry[name] = OperatorDict(
                    name, codegen=expr, algebra=self,
                    codegen_symbolcls=codegen_symbolcls or self.codegen_symbolcls,
                    printer=printer, func_printer=func_printer, wrapper=wrapper, values_asarray=values_asarray)
            return self.registry[name]

        # See if we are being called as @jit or @jit()
        if expr is None:  # Called as @jit()
            return partial(wrap, name=name, symbolic=symbolic)

        # Called as @jit
        return wrap(expr, name=name, symbolic=symbolic)

    def compile(self, expr=None, /, *mvs, symbolic=True, printer=None, func_printer=None, wrapper=None, values_asarray=None) -> CompiledExpression:
        """
        Compile a GA :code:`expr` with specific symbolic multivectors.
        For typical use cases you'll probably want to use :code:`Algebra.jit` instead, since that does not require you
        to provide the symbolical multivectors yourself, and it caches the resulting compiled functions so you don't have to track them yourself.
        However, the finer level of control of :code:`compile` is extremely powerful in certain use cases.
        For example, consider that we need to repeatedly rotate the unit vectors of our algebra in order to compute the evolution of the frame vectors::

            def rotate_blade(R, blade):
                return R >> blade

            alg = Algebra(3)
            R = alg.bireflection(...)  # Either a symbolic or a numeric rotation
            e1 = alg.vector(e1=1)
            e1_prime = rotate_blade(R, e1)

        By inspecting the code generated in :code:`alg.sw[R, e1].func` or by looking at its docstring
        :code:`alg.sw[R, e1].func.__doc__`, we find that the generated code uses 14 muls and 5 adds.
        (Mind you that the built-in :code:`sw` operator already benefits from CSE; without it the same
        expression needs 18 muls and 7 adds.)
        However :code:`Algebra.jit` can never use the runtime values of the multivector coefficients,
        whereas we know that we only want to rotate unit vectors.
        So we can use :code:`Algebra.compile` to generate an even more specialized function::

            rotation = alg.bireflection(name='R', symbolcls=alg.codegen_symbolcls)
            e1 = alg.vector(e1=1)
            rotate_e1 = alg.compile(rotate_blade, rotation, e1)  # compile the function for these specific mvs.
            e1_prime = rotate_e1(R, e1)

        If we look at :code:`rotate_e1` we find that it has only 9 muls and 5 adds!
        Compiling common operations involving constant multivectors such as frames can therefore be highly beneficial.

        Beware that :code:`rotate_e1` should not be used on non-unit vectors, so the onus falls on you to enure you only call it with appropriate vectors.
        """
        wrapper = wrapper or self.wrapper
        printer = printer or self.printer
        values_asarray = values_asarray or self.values_asarray
        func_printer = func_printer or self.func_printer
        if symbolic:
            compiled_expr = do_compile_symbolic(expr, *mvs, printer=printer, func_printer=func_printer, wrapper=wrapper, values_asarray=values_asarray)
        else:
            compiled_expr = do_compile(expr, *mvs, wrapper=wrapper, values_asarray=values_asarray)
        return compiled_expr

    def multivector(self, *args, **kwargs) -> MultiVector:
        """ Create a new :class:`~kingdon.multivector.MultiVector`. """
        return MultiVector(self, *args, **kwargs)

    def evenmv(self, *args, **kwargs) -> MultiVector:
        """ Create a new :class:`~kingdon.multivector.MultiVector` in the even subalgebra. """
        grades = tuple(filter(lambda x: x % 2 == 0, range(self.d + 1)))
        return MultiVector(self, *args, grades=grades, **kwargs)

    def oddmv(self, *args, **kwargs) -> MultiVector:
        """
        Create a new :class:`~kingdon.multivector.MultiVector` of odd grades.
        """
        grades = tuple(filter(lambda x: x % 2 == 1, range(self.d + 1)))
        return MultiVector(self, *args, grades=grades, **kwargs)

    def purevector(self, *args, grade, **kwargs) -> KVector:
        """
        Create a new k-vector of the desired grade.

        :param grade: Grade of the multivector to create.
        """
        if grade > len(self._kvectors):
            raise MultiVector(self, *args, grades=(grade,), **kwargs)
        return self._kvectors[grade](self, *args, **kwargs)

    def graph(self, *subjects, graph_widget=GraphWidget, **options):
        """
        The graph function outputs :code:`ganja.js` renders and is meant
        for use in jupyter notebooks. The syntax of the graph function will feel
        familiar to users of :code:`ganja.js`: all position arguments are considered
        as subjects to graph, while all keyword arguments are interpreted as options
        to :code:`ganja.js`'s :code:`Algebra.graph` method.

        Example usage:

        .. code-block ::

            alg.graph(
                0xD0FFE1, [A,B,C],
                0x224488, A, "A", B, "B", C, "C",
                lineWidth=3, grid=1, labels=1
            )

        Will create

        .. image :: ../docs/_static/graph_triangle.png
            :scale: 50%
            :align: center

        If a function is given to :code:`Algebra.graph` then it is called without arguments:

        .. code-block ::

            def graph_func():
                return [
                    0xD0FFE1, [A,B,C],
                    0x224488, A, "A", B, "B", C, "C"
                ]

            alg.graph(
                graph_func,
                lineWidth=3, grid=1, labels=1
            )

        This can be used to make animations in a manner identical to :code:`ganja.js` by making
        :code:`graph_func` depend on time and setting :code:`animate=True`.

        .. rubric:: Subjects

        The following types are accepted as positional arguments:

        - :class:`~kingdon.multivector.MultiVector`: rendered according to its grade and the
          algebra's signature, e.g. as a point, line, plane, circle, sphere, etc.
          A multivector with array-valued coefficients (e.g. :code:`numpy` arrays) is unpacked
          into the individual multivectors it represents.
        - :code:`int`: a hexadecimal color such as :code:`0x224488`, which sets the color of all
          subsequent subjects until the next color.
        - :code:`str`: a label, drawn at the position of the last drawn subject. Strings starting
          with :code:`"<"` are inserted verbatim as SVG, strings enclosed in :code:`$` are typeset
          as TeX (when a TeX renderer is loaded on the page), and strings starting with :code:`"_"` are not drawn but
          do advance the text cursor.
        - :code:`list` or :code:`tuple`: two multivectors are drawn as a line segment, three or more
          as a filled polygon. Lists may be nested, and may contain any of the types above.
        - :code:`Callable`: called without arguments, and the result is graphed. A single callable
          returning a list of subjects is typically used for animations; callables
          nested deeper in the input are re-evaluated in the same way.

        Multivectors passed directly as positional arguments (i.e. not nested inside a list) may be
        draggable: dragging them in the canvas writes the new coefficients back into the Python
        object in-place. In PGA only points (grade :math:`d-1`) are draggable, in conformal
        algebras (:code:`conformal=True`) only grade 1 elements are draggable.

        .. rubric:: Options

        All keyword arguments are passed on to :code:`ganja.js`'s :code:`Algebra.graph` directly,
        with the exception of :code:`camera`, :code:`up`, :code:`width`, :code:`height` and :code:`style`,
        which are (also) interpreted by :code:`kingdon`.
        As such, :code:`kingdon` automatically inherits all the options from :code:`ganja.js`.
        The most useful ones are mentioned in the list below.

        :param `*subjects`: The subjects to be graphed, see above.
        :param camera: A motor which places the camera at the desired viewpoint. Defaults to the
            identity motor.
        :param width: CSS width of the canvas, e.g. :code:`'600px'` or :code:`'100%'`.
            Defaults to :code:`'min( 100%, 1024px )'`.
        :param height: CSS height of the canvas, e.g. :code:`'400px'`. Defaults to :code:`'auto'`,
            in which case the aspect ratio of 16 / 6 determines the height.
        :param style: Dictionary of additional CSS properties (camelCased, as in JS) applied to the
            canvas, e.g. :code:`{'background': 'black', 'aspectRatio': '1 / 1'}`.
        :param animate: If truthy, the scene is redrawn every frame and all callables are
            re-evaluated, which is what drives animations. Defaults to :code:`False`, in which case
            the scene is only redrawn when a subject changes or the user interacts with it.
        :param grid: Draw a grid.
        :param labels: Label the grid lines.
        :param gridSize: Size of the grid in world units.
        :param gridFontSize: Scale of the grid labels.
        :param scale: Zoom level of the scene (2D/SVG rendering), default 1.
        :param lineWidth: Line width multiplier.
        :param pointRadius: Point radius multiplier.
        :param fontSize: Label size multiplier.
        :param conformal: Use the conformal renderer, needed for CGA.
        :param up: The 'up' (:math:`C`) function, a callable taking Euclidean coordinates and
            returning the corresponding element of the (larger) embedding space. Providing an `up` function
            invokes ganja's OPNS renderer, which can render any algebra, e.g. CGA, 2D CSGA, 3D CCGA,
            3D Mother Algebra. The function is traced symbolically and compiled to GLSL, so it must
            consist of algebraic operations only. See the teahouse for examples.
        :param ipns: Interpret the subjects as IPNS (dual) elements instead of OPNS.
        :param gl: Force the WebGL renderer (the default in 3D), which supports raytraced surfaces.
        :param spin: Rotate the camera continuously at the given rate (WebGL only).
        :param thresh: Threshold used by the raymarcher to decide when a surface is hit.
        :param `**options`: Any other option supported by :code:`ganja.js`'s
            :code:`Algebra.graph`, such as :code:`alpha`, :code:`cull`, :code:`noZ`,
            :code:`htmlText`, :code:`devicePixelRatio`, :code:`clip`, :code:`still`.

        :return: A :class:`~kingdon.graph.GraphWidget` displaying the scene with :code:`ganja.js`.
            The :meth:`~kingdon.graph.GraphWidget.update` method can be used to redraw an existing figure.
        """
        return graph_widget(
            algebra=self,
            raw_subjects=subjects,
            options=options,
        )

    def _blade2canon(self, basis_blade: str):
        """ Retrieve the canonical blade for a given blade, and the number of sign swaps required. """
        if basis_blade in self.canon2bin:
            return basis_blade, 0
        # if a generator isn't found, return a generator outside of the current space.
        bin = reduce(operator.or_, (self.canon2bin.get(f'e{i}', 2 ** self.d) for i in basis_blade[1:]))
        canon_blade = self.bin2canon.get(bin, False)
        if canon_blade:
            swaps, *_ = _swap_blades(basis_blade, '', target=canon_blade)
            return canon_blade, swaps
        return f'e{2 ** self.d}', 0

    def _swap_blades_bin(self, A: int, B: int):
        """
        Swap basis blades binary style. Not currently used because (surprisingly) this does not
        seem to be faster than the string manipulation version.
        """
        ab = A & B
        res = A ^ B
        if ab & ((1 << self.r) - 1):
            return [0, 0]

        t = A >> 1
        t ^= t >> 1
        t ^= t >> 2
        t ^= t >> 4
        t ^= t >> 8

        t &= B
        t ^= ab >> (self.p + self.r)
        t ^= t >> 16
        t ^= t >> 8
        t ^= t >> 4
        return [res, 1 - 2 * (27030 >> (t & 15) & 1)]

    def _bind_layout(self, MVType: type[MultiVector], name: str) -> dict:
        r"""
        Bind a layout to this algebra. If MVType defines a layout then this is straightforward,
        otherwise it has to be generated from the archetype.

        The archetype is a symbolic multivector, obtained by evaluating the GA expression
        that defines the type. Its coefficients are what we are after: a coefficient that
        came out numerical is a structural constant of the type, e.g. the :code:`1.0` a
        normalized point has on :math:`\mathbf{e}_0^*`, while anything else is a free component.
        """
        if hasattr(MVType, 'layout') and isinstance(MVType.layout, dict):
            layout = {}
            for blade, val in MVType.layout.items():
                if isinstance(blade, int): layout[blade] = val
                else:
                    # For fixed values we apply sign swaps to the value, for free values we store it in the key.
                    canon, swaps = self._blade2canon(blade)
                    sign = (-1 if swaps % 2 != 0 else 1)
                    if val == ...:
                        k = sign * self.canon2bin[canon]
                    else:
                        k = self.canon2bin[canon]
                        val = sign * val
                    layout[k] = val
            return layout

        archetype = MVType.archetype(self, name)
        def is_number(x):
            try:
                float(x); return True
            except (ValueError, TypeError):
                return False
        return {k: float(f) if is_number(f := str(v)) else ...
                for k, v in archetype.items()}

def _swap_blades(blade1: str, blade2: str, target: str = '') -> (int, str, str):
    """
    Compute the number of swaps of orthogonal vectors needed to pair the basis vectors. E.g. in
    ['1', '2', '3', '1', '2'] we need 3 swaps to get to ['1', '1', '2', '2', '3']. Pairs are also removed,
    in order to find the resulting blade; in the above example the result is ['3'].

    The output of the function is the number of swaps, the resulting blade indices, and the eliminated indices. E.g.

    .. code-block ::

            >>> _swap_blades('123', '12')
            3, '3', '12'
    """
    blade1 = list(blade1)
    swaps = 0
    eliminated = []
    for char in blade2:
        if char not in blade1:  # Move char from blade2 to blade1
            blade1.append(char)
            continue

        idx = blade1.index(char)
        swaps += len(blade1) - idx - 1
        blade1.remove(char)
        eliminated.append(char)

    if target:
        # Find the number of additional swaps needed to match the target.
        for i, char in enumerate(target):
            idx = blade1.index(char)
            blade1.insert(i, blade1.pop(idx))
            swaps += idx - i

    return swaps, ''.join(blade1), ''.join(eliminated)


class DefaultKeyDict(dict):
    """
    A lightweight dict subclass that behaves like a defaultdict
    but calls the factory function with the key as argument.
    """
    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        res = self[key] = self.factory(key)
        return res


@dataclass
class BladeDict(Mapping):
    """
    Dictionary of basis blades. Use getitem or getattr to retrieve a basis blade from this dict, e.g.::

        alg = Algebra(3, 0, 1)
        blade_dict = BladeDict(alg, lazy=True)
        blade_dict['e03']
        blade_dict.e03

    When `lazy=True`, the basis blade is only initiated when requested.
    This is done for performance in higher dimensional algebras.
    """
    algebra: Algebra
    lazy: bool = field(default=False)
    blades: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.lazy:
            # If not lazy, retrieve all blades once to force initiation.
            for blade in self.algebra.canon2bin: self[blade]
        else:
            self.grade(1)  # Initiate basis vectors only.

    def __getitem__(self, basis_blade):
        """ Blade must be in canonical form, e.g. 'e12'. """
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            raise AttributeError(f'{basis_blade} is not a valid basis blade.')
        basis_blade, swaps = self.algebra._blade2canon(basis_blade)
        if basis_blade not in self.blades:
            bin_blade = self.algebra.canon2bin[basis_blade]
            MVType, layout = resolve_layout(self.algebra._type_layouts, {bin_blade: 1})
            if self.algebra.graded:
                keysvalues = tuple((idx, int(bin_blade == idx)) if idx >= 0 else (-idx, -int(bin_blade == -idx))
                                     for idx, value in layout.items() if value == ...)
                keys, values = zip(*keysvalues) if keysvalues else ((), [])
                values = list(values)
            else:
                if layout.get(bin_blade) == ...:    keys, values = ((bin_blade,), [1])
                elif layout.get(-bin_blade) == ...: keys, values = ((bin_blade,), [-1])
                else:                               keys, values = ((), [])
            self.blades[basis_blade] = MVType.fromkeysvalues(self.algebra, keys=keys, values=values)
        return self.blades[basis_blade] if swaps % 2 == 0 else - self.blades[basis_blade]

    def __getattr__(self, blade):
        return self[blade]

    def __len__(self):
        return len(self.blades)

    def __iter__(self):
        return iter(self.blades)

    def grade(self, *grades) -> dict:
        """
        Return blades of grade `grades`.

        :param grades: tuple or ints, grades to select.
        """
        if len(grades) == 1 and isinstance(grades[0], tuple):
            grades = grades[0]

        return {(blade := self.algebra.bin2canon[k]): self[blade]
                for k in self.algebra.indices_for_grades(grades)}
