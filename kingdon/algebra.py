from itertools import combinations, product, chain, groupby
from functools import partial
from collections import Counter
from dataclasses import dataclass, field, fields
from collections.abc import Mapping, Callable
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
    codegen_polarity, codegen_unpolarity, codegen_hodge, codegen_unhodge
)
from kingdon.operator_dict import OperatorDict, UnaryOperatorDict, Registry
from kingdon.matrixreps import matrix_rep
from kingdon.multivector import MultiVector
from kingdon.polynomial import RationalPolynomial
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
    """
    p: int = 0
    q: int = 0
    r: int = 0
    d: int = field(init=False, repr=False, compare=False)  # Total number of dimensions
    signature: np.ndarray = field(default=None, compare=False)
    start_index: int = field(default=None, repr=False, compare=False)

    # Clever dictionaries that cache previously symbolically optimized lambda functions between elements.
    gp: OperatorDict = operation_field(metadata={'codegen': codegen_gp})  # geometric product
    sw: Registry = operation_field(metadata={'codegen': codegen_sw})  # conjugation
    cp: OperatorDict = operation_field(metadata={'codegen': codegen_cp})  # commutator product
    acp: OperatorDict = operation_field(metadata={'codegen': codegen_acp})  # anti-commutator product
    ip: OperatorDict = operation_field(metadata={'codegen': codegen_ip})  # inner product
    sp: OperatorDict = operation_field(metadata={'codegen': codegen_sp})  # Scalar product
    lc: OperatorDict = operation_field(metadata={'codegen': codegen_lc})  # left-contraction
    rc: OperatorDict = operation_field(metadata={'codegen': codegen_rc})  # right-contraction
    op: OperatorDict = operation_field(metadata={'codegen': codegen_op})  # exterior product
    rp: OperatorDict = operation_field(metadata={'codegen': codegen_rp})  # regressive product
    proj: Registry = operation_field(metadata={'codegen': codegen_proj})  # projection
    add: OperatorDict = operation_field(metadata={'codegen': codegen_add})  # add
    sub: OperatorDict = operation_field(metadata={'codegen': codegen_sub})  # sub
    div: OperatorDict = operation_field(metadata={'codegen': codegen_div})  # division
    # Unary operators
    inv: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_inv})  # inverse
    neg: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_neg})  # negate
    reverse: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_reverse})  # reverse
    involute: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_involute})  # grade involution
    conjugate: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_conjugate})  # clifford conjugate
    sqrt: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_sqrt})  # Square root
    polarity: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_polarity})
    unpolarity: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_unpolarity})
    hodge: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_hodge})
    unhodge: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_unhodge})
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
    _bin2canon_prettystr: dict = field(init=False, repr=False, compare=False)

    # Options for the algebra
    cse: bool = field(default=True, repr=False)  # Common Subexpression Elimination (CSE)
    graded: bool = field(default=False, repr=False)  # If true, precompute products per grade.

    # Codegen & call customization.
    # Wrapper function applied to the codegen generated functions.
    wrapper: Callable = field(default=None, repr=False, compare=False)
    # The symbol class used in codegen. By default, use our own fast RationalPolynomial class.
    codegen_symbolcls: object = field(default=RationalPolynomial.fromname, repr=False, compare=False)
    # This simplify func is applied to every component after a symbolic expression is called, to simplify and filter by.
    simp_func: Callable = field(default=lambda v: v if not isinstance(v, sympy.Expr) else sympy.simplify(sympy.expand(v)), repr=False, compare=False)

    signs: dict = field(init=False, repr=False, compare=False)
    cayley: dict = field(init=False, repr=False, compare=False)
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

        # Setup mapping from binary to canonical string rep and vise versa
        self.bin2canon = {
            eJ: 'e' + ''.join(hex(num + self.start_index - 1)[2:] for ei in range(0, self.d) if (num := (eJ & 2**ei).bit_length()))
            for eJ in range(2 ** self.d)
        }
        self.canon2bin = dict(sorted({c: b for b, c in self.bin2canon.items()}.items(), key=lambda x: (len(x[0]), x[0])))
        def pretty_blade(blade):
            if blade == 'e':
                return '1'
            blade = '𝐞' + blade[1:]
            for old, new in tuple(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉")):
                blade = blade.replace(old, new)
            return blade
        self._bin2canon_prettystr = {k: pretty_blade(v) for k, v in self.bin2canon.items()}

        self.swaps, self.signs, self.cayley = self._prepare_signs_and_cayley()

        # Blades are not precomputed for algebras larger than 6D.
        self.blades = BladeDict(algebra=self, lazy=self.d > 6)

        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

        # Prepare OperatorDict's
        self.registry = {f.name: f.type(name=f.name, codegen=f.metadata['codegen'], algebra=self)
                         for f in fields(self) if 'codegen' in f.metadata}
        for name, operator_dict in self.registry.items():
            setattr(self, name, operator_dict)

    def __len__(self):
        return 2 ** self.d

    @cached_property
    def indices_for_grade(self):
        """
        Mapping from the grades to the indices for that grade. E.g. in 2D VGA, this returns

        .. code-block ::

            {0: (0,), 1: (1, 2), 2: (3,)}
        """
        return {length - 1: tuple(self.canon2bin[blade] for blade in blades)
                for length, blades in groupby(self.canon2bin, key=len)}

    @cached_property
    def indices_for_grades(self):
        """
        Mapping from a sequence of grades to the corresponding indices.
        E.g. in 2D VGA, this returns

        .. code-block ::

            {(): (), (0,): (0,), (1,): (1, 2), (2,): (3,), (0, 1): (0, 1, 2),
             (0, 2): (0, 3), (1, 2): (1, 2, 3), (0, 1, 2): (0, 1, 2, 3)}
        """
        all_grade_combs = chain(*(combinations(range(0, self.d + 1), r=j) for j in range(0, len(self) + 1)))
        return {comb: sum((self.indices_for_grade[grade] for grade in comb), ())
                for comb in all_grade_combs}

    @cached_property
    def matrix_basis(self):
        return matrix_rep(self.p, self.q, self.r)

    @cached_property
    def frame(self) -> list:
        """
        The set of orthogonal basis vectors, :math:`\{ e_i \}`. Note that for a frame linear independence suffices,
        but we already have orthogonal basis vectors so why not use those?
        """
        return [self.blades[self.bin2canon[2**j]] for j in range(0, self.d)]

    @cached_property
    def reciprocal_frame(self) -> list:
        """
        The reciprocal frame is a set of vectors :math:`\{ e^i \}` that satisfies
        :math:`e^i \cdot e_j = \delta^i_j` with the frame vectors :math:`\{ e_i \}`.
        """
        return [v.inv() for v in self.frame]

    def _prepare_signs_and_cayley(self):
        """
        Prepares two dicts whose keys are two basis-blades (in binary rep) and the result is either
        just the sign (1, -1, 0) of the corresponding multiplication, or the full result.
        The full result is essentially the Cayley table, if printed as a table.

        E.g. in :math:`\mathbb{R}_2`, sings[(0b11, 0b11)] = -1.
        """
        cayley = {}
        signs = np.zeros((len(self), len(self)), dtype=int)
        swaps_arr = np.zeros((len(self), len(self)), dtype=int)
        # swap_dict = {}
        for eI, eJ in product(self.canon2bin, repeat=2):
            # Compute the number of swaps of orthogonal vectors needed to order the basis vectors.
            prod = list(eI[1:] + eJ[1:])
            swaps = _sort_product(prod) if len(prod) else 0
            swaps_arr[self.canon2bin[eI], self.canon2bin[eJ]] = swaps

            # Remove even powers of basis-vectors.
            sign = -1 if swaps % 2 else 1
            count = Counter(prod)
            for key, value in count.items():
                if value // 2:
                    sign *= self.signature[int(key, base=16) - self.start_index]
                count[key] = value % 2
            signs[self.canon2bin[eI], self.canon2bin[eJ]] = sign

            # Make the Cayley table.
            if sign:
                prod = ''.join(key*value for key, value in count.items())
                sign = '-' if sign == -1 else ''
                cayley[eI, eJ] = f'{sign}e{prod}'
            else:
                cayley[eI, eJ] = f'0'
        return swaps_arr, signs, cayley

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
        :param `**options`: Options passed to :code:`ganja.js`'s :code:`Algebra.graph`.
        """
        return graph_widget(
            algebra=self,
            raw_subjects=subjects,
            options=options,
        )


def _sort_product(prod):
    """
    Compute the number of swaps of orthogonal vectors needed to order the basis vectors. E.g. in
    ['1', '2', '3', '1', '2'] we need 3 swaps to get to ['1', '1', '2', '2', '3'].

    Changes the input list! This is by design.
    """
    swaps = 0
    if len(prod) > 1:
        prev_swap = 0
        while True:
            for i in range(len(prod) - 1):
                if prod[i] > prod[i + 1]:
                    swaps += 1
                    prod[i], prod[i + 1] = prod[i + 1], prod[i]
            if prev_swap == swaps:
                break
            else:
                prev_swap = swaps
    return swaps


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

    def __getitem__(self, blade):
        """ Blade must be in canonical form, e.g. 'e12'. """
        if blade not in self.blades:
            bin_blade = self.algebra.canon2bin[blade]
            if self.algebra.graded:
                g = format(bin_blade, 'b').count('1')
                indices = self.algebra.indices_for_grade[g]
                self.blades[blade] = self.algebra.multivector(values=[int(bin_blade == i) for i in indices], grades=(g,))
            else:
                self.blades[blade] = MultiVector.fromkeysvalues(self.algebra, keys=(bin_blade,), values=[1])
        return self.blades[blade]

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
                for k in self.algebra.indices_for_grades[grades]}
