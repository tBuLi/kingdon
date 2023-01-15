from itertools import combinations, product, chain, groupby
from functools import partial
from collections import Counter
from dataclasses import dataclass, field, fields
import json
try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        return property(lru_cache()(func))

import numpy as np
from IPython.display import Javascript, display

from kingdon.codegen import (
    codegen_gp, codegen_sw, codegen_cp, codegen_ip, codegen_op, codegen_div,
    codegen_rp, codegen_acp, codegen_proj, codegen_sp, codegen_lc, codegen_inv,
    codegen_rc, codegen_normsq,
    codegen_outerexp, codegen_outersin, codegen_outercos, codegen_outertan,
)
from kingdon.operator_dict import OperatorDict, UnaryOperatorDict
from kingdon.matrixreps import matrix_rep
from kingdon.multivector_json import MultiVectorEncoder
from kingdon.multivector import MultiVector

# from kingdon.module_builder import predefined_modules

operation_field = partial(field, default_factory=dict, init=False, repr=False, compare=False)


@dataclass
class Algebra:
    """
    A Geometric (Clifford) algebra with :code:`p` positive dimensions,
    :code:`q` negative dimensions, and :code:`r` null dimensions.

    :param p:  number of positive dimensions.
    :param q:  number of negative dimensions.
    :param r:  number of null dimensions.
    :param cse: If :code:`True` (default), attempt Common Subexpression Elimination (CSE)
        on symbolically optimized expressions.
    :param numba: If :code:`True` (default), use numba.njit to just-in-time compile expressions.
    :param graded: If :code:`True` (default), perform binary and unary operations on a graded basis.
        This will still be more sparse than computing with a full multivector, but not as sparse as possible.
        It does however, fastly reduce the number of possible expressions that have to be symbolically optimized.
    """
    p: int = 0
    q: int = 0
    r: int = 0
    d: int = field(init=False, repr=False, compare=False)  # Total number of dimensions
    signature: np.ndarray = field(default=None, compare=False)
    start_index: int = field(default=None, repr=False, compare=False)

    # Clever dictionaries that cache previously symbolically optimized lambda functions between elements.
    gp: OperatorDict = operation_field(metadata={'codegen': codegen_gp})  # geometric product
    conj: OperatorDict = operation_field(metadata={'codegen': codegen_sw})  # conjugation
    cp: OperatorDict = operation_field(metadata={'codegen': codegen_cp})  # commutator product
    acp: OperatorDict = operation_field(metadata={'codegen': codegen_acp})  # anti-commutator product
    ip: OperatorDict = operation_field(metadata={'codegen': codegen_ip})  # inner product
    sp: OperatorDict = operation_field(metadata={'codegen': codegen_sp})  # Scalar product
    lc: OperatorDict = operation_field(metadata={'codegen': codegen_lc})  # left-contraction
    rc: OperatorDict = operation_field(metadata={'codegen': codegen_rc})  # right-contraction
    op: OperatorDict = operation_field(metadata={'codegen': codegen_op})  # exterior product
    rp: OperatorDict = operation_field(metadata={'codegen': codegen_rp})  # regressive product
    proj: OperatorDict = operation_field(metadata={'codegen': codegen_proj})  # projection
    div: OperatorDict = operation_field(metadata={'codegen': codegen_div})  # division
    inv: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_inv})  # inverse
    normsq: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_normsq})  # norm squared
    outerexp: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outerexp})
    outersin: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outersin})
    outercos: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outercos})
    outertan: UnaryOperatorDict = operation_field(metadata={'codegen': codegen_outertan})

    # Mappings from binary to canonical reps. e.g. 0b01 = 1 <-> 'e1', 0b11 = 3 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False, compare=False)
    bin2canon: dict = field(init=False, repr=False, compare=False)
    _bin2canon_prettystr: dict = field(init=False, repr=False, compare=False)

    # Options for the algebra
    cse: bool = field(default=True)  # Common Subexpression Elimination (CSE)
    precompute: str = field(default='none')  # Precompute (common) products. Options: 'none' (default), 'all', 'common'.
    numba: bool = field(default=False)  # Enable numba just-in-time compilation
    graded: bool = field(default=False)  # If true, precompute products per grade.
    simplify: bool = field(default=True)  # If true, perform symbolic simplification

    signs: dict = field(init=False, repr=False, compare=False)
    cayley: dict = field(init=False, repr=False, compare=False)
    blades: dict = field(init=False, repr=False, compare=False)
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
            eJ: 'e' + ''.join(str(num + self.start_index - 1) for ei in range(0, self.d) if (num := (eJ & 2**ei).bit_length()))
            for eJ in range(2 ** self.d)
        }
        self.canon2bin = dict(sorted({c: b for b, c in self.bin2canon.items()}.items(), key=lambda x: (len(x[0]), x[0])))
        def pretty_blade(blade):
            if blade == 'e':
                return '1'
            for old, new in tuple(zip("e0123456789", "𝐞₀₁₂₃₄₅₆₇₈₉")):
                blade = blade.replace(old, new)
            return blade
        self._bin2canon_prettystr = {k: pretty_blade(v) for k, v in self.bin2canon.items()}

        self.swaps, self.signs, self.cayley = self._prepare_signs_and_cayley()

        self.blades = {k: MultiVector.fromkeysvalues(self, keys=(v,), values=(1,))
                       for k, v in self.canon2bin.items()}
        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

        # Prepare OperatorDict's
        operators = (f for f in fields(self) if 'codegen' in f.metadata)
        for f in operators:
            setattr(self, f.name, f.type(name=f.name, codegen=f.metadata['codegen'], algebra=self))

    def __len__(self):
        return 2 ** self.d

    @cached_property
    def indices_for_grade(self):
        """
        Mapping from the grades to the indices for that grade. E.g. in 2D VGA, this returns

        .. code-block ::

            {0: (0,), 1: (1, 2), 2: (3,)}
        """
        key = lambda i: bin(i).count('1')
        sorted_inds = sorted(range(len(self)), key=key)
        return {grade: tuple(inds) for grade, inds in groupby(sorted_inds, key=key)}

    @cached_property
    def indices_for_grades(self):
        """
        Mapping from a sequence of grades to the corresponding indices.
        E.g. in 2D VGA, this returns

        .. code-block ::

            {(0,): (0,), (1,): (1, 2), (2,): (3,), (0, 1): (0, 1, 2),
             (0, 2): (0, 3), (1, 2): (1, 2, 3), (0, 1, 2): (0, 1, 2, 3)}
        """
        all_grade_combs = chain(*(combinations(range(0, self.d + 1), r=j) for j in range(1, len(self) + 1)))
        return {comb: sum((self.indices_for_grade[grade] for grade in comb), ())
                for comb in all_grade_combs}

    @cached_property
    def matrix_basis(self):
        return matrix_rep(self.p, self.q, self.r)

    def _prepare_signs_and_cayley(self):
        """
        Prepares two dicts whose keys are two basis-blades in (binary rep) and the result is either
        just the sign (1, -1, 0) of the corresponding multiplication, or the full result.
        The full result is essentially the Cayley table, if printed as a table.

        E.g. in :math:`\mathbb{R}_2`, sings[(0b11, 0b11)] = -1.
        """
        cayley = {}
        signs = np.zeros((len(self), len(self)), dtype=int)
        swaps_arr = np.zeros((len(self), len(self)), dtype=int)
        # swap_dict = {}
        for eI, eJ in product(self.canon2bin, repeat=2):
            prod = eI[1:] + eJ[1:]
            # Compute the number of swaps of orthogonal vectors needed to order the basis vectors.
            swaps = 0
            if len(prod) > 1:
                prev_swap = 0
                while True:
                    for i in range(len(prod) - 1):
                        if prod[i] > prod[i + 1]:
                            swaps += 1
                            prod = prod[:i] + prod[i + 1] + prod[i] + prod[i+2:]
                    if prev_swap == swaps:
                        break
                    else:
                        prev_swap = swaps
            swaps_arr[self.canon2bin[eI], self.canon2bin[eJ]] = swaps

            # Remove even powers of basis-vectors.
            sign = -1 if swaps % 2 else 1
            count = Counter(prod)
            for key, value in count.items():
                if value // 2:
                    sign *= self.signature[int(key) - self.start_index]
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

    def graph(self, *subjects, **options):
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

        Not all features of :code:`ganja.js` are supported yet. Most notably,
        only static graphs can be made. While ganja also accepts functions as
        input, this syntax is not currently supported in Kingdon.

        :param `*subjects`: Subjects to be graphed.
            Can be strings, hexadecimal colors, (lists of) MultiVector.
        :param `**options`: Options passed to :code:`ganja.js`'s :code:`Algebra.graph`.
        """
        # Flatten multidimensional multivectors
        flat_subjects = []
        for subject in subjects:
            if isinstance(subject, MultiVector) and len(subject.shape()) > 1:
                flat_subjects.extend(subject.itermv())
            else:
                flat_subjects.append(subject)

        json_subjects = json.dumps(flat_subjects, cls=MultiVectorEncoder)

        cayley_table = [[s if (s := self.cayley[eJ, eI])[-1] != 'e' else f"{s[:-1]}1"
                         for eI in self.canon2bin]
                        for eJ in self.canon2bin]
        cayley_table = json.dumps(cayley_table)
        metric = json.dumps(list(self.signature), cls=MultiVectorEncoder)

        src = f"""
        fetch("https://enki.ws/ganja.js/ganja.js")
        .then(x=>x.text())
        .then(ganja=>{{

          var f = new Function("module",ganja);
          var module = {{exports:{{}}}};
          f(module);
          var Algebra = module.exports;

          var canvas = Algebra({{metric:{metric}, Cayley:{cayley_table}}},()=>{{
              var data = {json_subjects}.map(x=>x.length=={len(self)}?new Element(x):x);
              return this.graph(data, {options})
          }})

          element.append(canvas)

        }})
        """
        display(Javascript(src))
