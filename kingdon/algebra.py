import operator
import re
from itertools import combinations, product, chain
from functools import partial, reduce
from collections import Counter
from dataclasses import dataclass, field, fields
from collections.abc import Mapping, Callable
from typing import List, Tuple

try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        return property(lru_cache()(func))

import numpy as np
import sympy

from kingdon.codegen import (
    codegen_gp, codegen_sw, codegen_cp, codegen_ip, codegen_op, codegen_div,
    codegen_rp, codegen_acp, codegen_proj, codegen_sp, codegen_lc, codegen_inv,
    codegen_rc, codegen_normsq, codegen_add, codegen_neg, codegen_reverse,
    codegen_involute, codegen_conjugate, codegen_sub, codegen_sqrt,
    codegen_outerexp, codegen_outersin, codegen_outercos, codegen_outertan,
    codegen_polarity, codegen_unpolarity, codegen_hodge, codegen_unhodge,
)
from kingdon.operator_dict import OperatorDict, UnaryOperatorDict, Registry, do_operation, resolve_and_expand
from kingdon.polynomial import mathstr
from kingdon.matrixreps import matrix_rep
from kingdon.multivector import MultiVector
from kingdon.graph import GraphWidget

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
    :param start_index: Optionally set the start index of the dimensions. For PGA this defualts to `0`, otherwise `1`.
    :param basis: Custom basis order, e.g. `["e", "e1", "e2", "e0", "e20", "e01", "e12", "e012"]` for 2DPGA.
    :param cse: If :code:`True` (default), attempt Common Subexpression Elimination (CSE)
        on symbolically optimized expressions.
    :param graded: If :code:`True` (default is :code:`False`), perform binary and unary operations on a graded basis.
        This will still be more sparse than computing with a full multivector, but not as sparse as possible.
        It does however, vastly reduce the number of possible expressions that have to be symbolically optimized.
    :param simplify: If :code:`True` (default), we attempt to simplify as much as possible. Setting this to
        :code:`False` will reduce the number of calls to simplify. However, it seems that :code:`True` is still faster,
        probably because it keeps sympy expressions from growing too large, which makes both symbolic computations and
        printing into a python function slower.
    :param wrapper: A function that is always applied to the generated functions as a decorator. For example,
        using :code:`numba.njit` as a wrapper will ensure that all kingdon code is jitted using numba.
    :param codegen_symbolcls: The symbol class used during codegen. By default, this is our own fast
        :class:`~kingdon.polynomial.RationalPolynomial` class.
    :param simp_func: This function is applied as a filter function to every multivector coefficient.
    :param pretty_blade: character to use for basis blades when pretty printing to string. Default is 𝐞.
    :param large: if true this is considered a large algebra. This means various cashing options are removed to save
        memory, and codegen is replaced by direct computation since codegen is very resource intensive for big
        expressions. By default, algebras of :math:`d > 6` are considered large, but the user can override this setting
        because also in large algebras it is still true that the generated code will perform order(s) of magnitude
        better than direct computation.
    """
    p: int = 0
    q: int = 0
    r: int = 0
    d: int = field(init=False, repr=False, compare=False)  # Total number of dimensions
    signature: np.ndarray = field(default=None, compare=False)
    start_index: int = field(default=None, repr=False, compare=False)
    basis: List[str] = field(repr=False, default_factory=list)

    # Clever dictionaries that cache previously symbolically optimized lambda functions between elements.
    gp: OperatorDict = operation_field(metadata={'codegen': codegen_gp, 'codegen_symbolcls': mathstr})  # geometric product
    sw: OperatorDict = operation_field(metadata={'codegen': codegen_sw})  # conjugation
    cp: OperatorDict = operation_field(metadata={'codegen': codegen_cp, 'codegen_symbolcls': mathstr})  # commutator product
    acp: OperatorDict = operation_field(metadata={'codegen': codegen_acp, 'codegen_symbolcls': mathstr})  # anti-commutator product
    ip: OperatorDict = operation_field(metadata={'codegen': codegen_ip, 'codegen_symbolcls': mathstr})  # inner product
    sp: OperatorDict = operation_field(metadata={'codegen': codegen_sp, 'codegen_symbolcls': mathstr})  # Scalar product
    lc: OperatorDict = operation_field(metadata={'codegen': codegen_lc, 'codegen_symbolcls': mathstr})  # left-contraction
    rc: OperatorDict = operation_field(metadata={'codegen': codegen_rc, 'codegen_symbolcls': mathstr})  # right-contraction
    op: OperatorDict = operation_field(metadata={'codegen': codegen_op, 'codegen_symbolcls': mathstr})  # exterior product
    rp: OperatorDict = operation_field(metadata={'codegen': codegen_rp, 'codegen_symbolcls': mathstr})  # regressive product
    proj: OperatorDict = operation_field(metadata={'codegen': codegen_proj})  # projection
    add: OperatorDict = operation_field(metadata={'codegen': codegen_add, 'codegen_symbolcls': mathstr})  # add
    sub: OperatorDict = operation_field(metadata={'codegen': codegen_sub, 'codegen_symbolcls': mathstr})  # sub
    div: OperatorDict = operation_field(metadata={'codegen': codegen_div})  # division
    # Unary operators
    inv: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_inv})  # inverse
    neg: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_neg, 'codegen_symbolcls': mathstr})  # negate
    reverse: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_reverse, 'codegen_symbolcls': mathstr})  # reverse
    involute: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_involute, 'codegen_symbolcls': mathstr})  # grade involution
    conjugate: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_conjugate, 'codegen_symbolcls': mathstr})  # clifford conjugate
    sqrt: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_sqrt})  # Square root
    polarity: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_polarity})
    unpolarity: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_unpolarity})
    hodge: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_hodge, 'codegen_symbolcls': mathstr})
    unhodge: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_unhodge, 'codegen_symbolcls': mathstr})
    normsq: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_normsq})  # norm squared
    outerexp: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outerexp})
    outersin: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outersin})
    outercos: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outercos})
    outertan: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outertan})
    registry: dict = field(default_factory=dict, repr=False, compare=False)  # Dict of all operator dicts. Should be extended using Algebra.register
    numspace: dict = field(default_factory=dict, repr=False, compare=False)  # Namespace for numerical functions

    # Mappings from binary to canonical reps. e.g. 0b01 = 1 <-> 'e1', 0b11 = 3 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False, compare=False)
    bin2canon: dict = field(init=False, repr=False, compare=False)

    # Options for the algebra
    cse: bool = field(default=True, repr=False)  # Common Subexpression Elimination (CSE)
    graded: bool = field(default=False, repr=False)  # If true, precompute products per grade.
    pretty_blade: str = field(default='𝐞', repr=False, compare=False)
    pretty_digits: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # TODO: this can be defined outside Algebra
    large: bool = field(default=None, repr=False, compare=False)

    # Codegen & call customization.
    # Wrapper function applied to the codegen generated functions.
    wrapper: Callable = field(default=None, repr=False, compare=False)
    # The symbol class used in codegen. By default, use our own fast RationalPolynomial class.
    codegen_symbolcls: object = field(default=None, repr=False, compare=False)
    # This simplify func is applied to every component after a symbolic expression is called, to simplify and filter by.
    simp_func: Callable = field(default=lambda v: v if not isinstance(v, sympy.Expr) else sympy.simplify(sympy.expand(v)), repr=False, compare=False)

    signs: dict = field(init=False, repr=False, compare=False)
    blades: "BladeDict" = field(init=False, repr=False, compare=False)
    pss: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.signature is not None:
            counts = Counter(self.signature)
            self.p, self.q, self.r = counts[1], counts[-1], counts[0]
            if self.p + self.q + self.r != len(self.signature):
                raise TypeError('Unsupported signature.')
            self.signature = np.array(self.signature)
        else:
            if self.r == 1:  # PGA, so put r first.
                self.signature = np.array([0] * self.r + [1] * self.p + [-1] * self.q)
            else:
                self.signature = np.array([1] * self.p + [-1] * self.q + [0] * self.r)

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

        self.signs = self._prepare_signs()

        if self.large is None:
            self.large = self.d > 6
        # Blades are not precomputed for large algebras.
        self.blades = BladeDict(algebra=self, lazy=self.large)

        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

        if self.large:
            self.registry = {f.name: self.wrapper(resolve_and_expand(partial(do_operation, codegen=codegen, algebra=self)))
                                     if self.wrapper else resolve_and_expand(partial(do_operation, codegen=codegen, algebra=self))
                             for f in fields(self) if (codegen := f.metadata.get('codegen'))}
        else:
            # Prepare OperatorDict's
            self.registry = {f.name: f.type(name=f.name, algebra=self, **f.metadata)
                             for f in fields(self) if 'codegen' in f.metadata}
        for name, op in self.registry.items():
            setattr(self, name, op)

    @classmethod
    def fromname(cls, name: str, **kwargs):
        """
        Initialize a well known algebra by its name. Options are 2DPGA, 3DPGA, and STAP.
        This uses sensible ordering of the basis vectors in the basis blades to avoid minus superfluous signs.
        """
        if name == '2DPGA':
            basis = ["e", "e1", "e2", "e0", "e20", "e01", "e12", "e012"]
            return cls(2, 0, 1, basis=basis, **kwargs)
        elif name == '3DPGA':
            basis = ["e", "e1", "e2", "e3", "e0",
                     "e01", "e02", "e03", "e12", "e31", "e23",
                     "e032", "e013", "e021", "e123", "e0123"]
            return cls(3, 0, 1, basis=basis, **kwargs)
        elif name == 'STAP':
            basis = ["e", "e0", "e1", "e2", "e3", "e4",
                     "e01", "e02", "e03", "e40", "e12", "e31", "e23", "e41", "e42", "e43",
                     "e234", "e314", "e124", "e123", "e014", "e024", "e034", "e032", "e013", "e021",
                     "e0324", "e0134", "e0214", "e0123", "e1234", "e01234"]
            return cls(3, 1, 1, basis=basis, **kwargs)
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
        """
        return (sum(2**bin for bin in bins) for bins in combinations(range(self.d), r=grade))

    def indices_for_grades(self, grades: Tuple[int]):
        """
        Function that returns a generator for all the indices from a sequence of grades.
        E.g. in 2D VGA, this returns

        .. code-block ::

            >>> alg = Algebra(2)
            >>> tuple(alg.indices_for_grades((1, 2)))
            (1, 2, 3)
        """
        return (chain.from_iterable(self.indices_for_grade(grade) for grade in sorted(grades)))

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

    def _prepare_signs(self):
        r"""
        Prepares a dict whose keys are a pair of basis-blades (in binary rep) and the
        result is the sign (1, -1, 0) of the corresponding multiplication.

        E.g. in :math:`\mathbb{R}_2`, sings[(0b11, 0b11)] = -1.
        """
        signs = {}

        def _compute_sign(bin_pair, canon_pair=None):
            I, J = bin_pair
            if not canon_pair:
                canon_pair = self.bin2canon[I], self.bin2canon[J]
            eI, eJ = canon_pair
            # Compute the number of swaps of orthogonal vectors needed to order the basis vectors.
            swaps, prod, eliminated = _swap_blades(eI[1:], eJ[1:], self.bin2canon[I ^ J][1:])

            # Remove even powers of basis-vectors.
            sign = -1 if swaps % 2 else 1
            for key in eliminated:
                sign *= self.signature[int(key, base=len(self.pretty_digits)) - self.start_index]
            return sign

        if self.d > 6:
            return DefaultKeyDict(_compute_sign)

        for (eI, I), (eJ, J) in product(self.canon2bin.items(), repeat=2):
            signs[I, J] = _compute_sign((I, J), (eI, eJ))

        return signs

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
        Register a function with the algebra to optimize its execution times.

        The function must be a valid GA expression, not an arbitrary python function.

        Example:

        .. code-block ::

            @alg.register
            def myexpr(a, b):
                return a @ b

            @alg.register(symbolic=True)
            def myexpr(a, b):
                return a @ b

        With default settings, the decorator will ensure that every GA unary or binary
        operator is replaced by the corresponding numerical function, and produces
        numerically much more performant code. The speed up is particularly notible when
        e.g. `self.wrapper=numba.njit`, because then the cost for all the python glue surrounding
        the actual computation has to be paid only once.

        When `symbolic=True` the expression is symbolically optimized before being turned
        into a numerical function. Beware that symbolic optimization of longer expressions
        (currently) takes exorbitant amounts of time, and often isn't worth it if the end
        goal is numerical computation.

        :param expr: Python function of a valid kingdon GA expression.
        :param name: (optional) name by which the function will be known to the algebra.
            By default, this is the `expr.__name__`.
        :param symbolic: (optional) If true, the expression is symbolically optimized.
            By default this is False, given the cost of optimizing large expressions.
        """
        def wrap(expr, name=None, symbolic=False):
            if name is None:
                name = expr.__name__

            if not symbolic:
                self.registry[expr] = Registry(name, codegen=expr, algebra=self)
            else:
                self.registry[expr] = OperatorDict(name, codegen=expr, algebra=self)
            return self.registry[expr]

        # See if we are being called as @register or @register()
        if expr is None:
            # Called as @register()
            return partial(wrap, name=name, symbolic=symbolic)

        # Called as @register
        return wrap(expr, name=name, symbolic=symbolic)

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
        (There is technically no such thing as an odd subalgebra, but
        otherwise this is similar to :class:`~kingdon.algebra.Algebra.evenmv`.)
        """
        grades = tuple(filter(lambda x: x % 2 == 1, range(self.d + 1)))
        return MultiVector(self, *args, grades=grades, **kwargs)

    def purevector(self, *args, grade, **kwargs) -> MultiVector:
        """
        Create a new :class:`~kingdon.multivector.MultiVector` of a specific grade.

        :param grade: Grade of the mutivector to create.
        """
        return MultiVector(self, *args, grades=(grade,), **kwargs)

    def scalar(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=0, **kwargs)

    def vector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=1, **kwargs)

    def bivector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=2, **kwargs)

    def trivector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=3, **kwargs)

    def quadvector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=4, **kwargs)

    def pseudoscalar(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=self.d - 0, **kwargs)

    def pseudovector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=self.d - 1, **kwargs)

    def pseudobivector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=self.d - 2, **kwargs)

    def pseudotrivector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=self.d - 3, **kwargs)

    def pseudoquadvector(self, *args, **kwargs) -> MultiVector:
        return self.purevector(*args, grade=self.d - 4, **kwargs)

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

        If a function is given to :code:`Algebra.graph` then it is called without arguments.
        This can be used to make animations in a manner identical to :code:`ganja.js`.

        Example usage:

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

        :param `*subjects`: Subjects to be graphed.
            Can be strings, hexadecimal colors, (lists of) MultiVector, (lists of) callables.
        :param camera: [optional] a motor that places the camera at the desired viewpoint.
        :param up: [optional] the 'up' (C) function that takes a Euclidean point and casts it into a larger
            embedding space. This will invoke ganja's OPNS renderer, which can be used to render any algebra.
            Examples include 2D CSGA, 3D CCGA, 3D Mother Algebra, etc. See the teahouse for examples.
        :param `**options`: Other options passed to :code:`ganja.js`'s :code:`Algebra.graph`.

        """
        return graph_widget(
            algebra=self,
            raw_subjects=subjects,
            options=options,
        )

    def _blade2canon(self, basis_blade: str):
        """ Retrieve the canonical blade for a given blade, and the number of sing swaps required. """
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
        Swap basis blades binary style. Not currently used because (suprinsingly) this does not
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

    def __getitem__(self, basis_blade):
        """ Blade must be in canonical form, e.g. 'e12'. """
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            raise AttributeError(f'{basis_blade} is not a valid basis blade.')
        basis_blade, swaps = self.algebra._blade2canon(basis_blade)
        if basis_blade not in self.blades:
            bin_blade = self.algebra.canon2bin[basis_blade]
            if self.algebra.graded:
                g = format(bin_blade, 'b').count('1')
                indices = self.algebra.indices_for_grade(g)
                self.blades[basis_blade] = self.algebra.multivector(values=[int(bin_blade == i) for i in indices], grades=(g,))
            else:
                self.blades[basis_blade] = MultiVector.fromkeysvalues(self.algebra, keys=(bin_blade,), values=[1])
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
