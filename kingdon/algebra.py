from itertools import combinations, product, chain, groupby
from functools import partial
from collections import Counter
from dataclasses import dataclass, field, fields
try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        return property(lru_cache()(func))

import numpy as np

from kingdon.codegen import (
    codegen_gp, codegen_conj, codegen_cp, codegen_ip, codegen_op, codegen_div,
    codegen_rp, codegen_acp, codegen_proj, codegen_sp, codegen_lc, codegen_inv,
    codegen_rc, codegen_normsq
)
from kingdon.operator_dict import OperatorDict
from kingdon.matrixreps import matrix_rep
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
    signature: list = field(init=False, repr=False, compare=False)

    # Clever dictionaries that cache previously symbolically optimized lambda functions between elements.
    gp: OperatorDict = operation_field(metadata={'codegen': codegen_gp})  # geometric product dict
    conj: OperatorDict = operation_field(metadata={'codegen': codegen_conj})  # conjugation dict
    cp: OperatorDict = operation_field(metadata={'codegen': codegen_cp})  # commutator product dict
    acp: OperatorDict = operation_field(metadata={'codegen': codegen_acp})  # anti-commutator product dict
    ip: OperatorDict = operation_field(metadata={'codegen': codegen_ip})  # inner product dict
    sp: OperatorDict = operation_field(metadata={'codegen': codegen_sp})  # Scalar product dict
    lc: OperatorDict = operation_field(metadata={'codegen': codegen_lc})  # left-contraction
    rc: OperatorDict = operation_field(metadata={'codegen': codegen_rc})  # right-contraction
    op: OperatorDict = operation_field(metadata={'codegen': codegen_op})  # exterior product dict
    rp: OperatorDict = operation_field(metadata={'codegen': codegen_rp})  # regressive product dict
    proj: OperatorDict = operation_field(metadata={'codegen': codegen_proj})  # projection dict
    div: OperatorDict = operation_field(metadata={'codegen': codegen_div})  # division dict
    inv: OperatorDict = operation_field(metadata={'codegen': codegen_inv})  # inverse dict
    normsq: OperatorDict = operation_field(metadata={'codegen': codegen_normsq})  # norm squared dict

    # Mappings from binary to canonical reps. e.g. 0b01 = 1 <-> 'e1', 0b11 = 3 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False, compare=False)
    bin2canon: dict = field(init=False, repr=False, compare=False)

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
        self.d = self.p + self.q + self.r
        self.signature = np.array([1]*self.p + [-1]*self.q + [0]*self.r)

        # Setup mapping from binary to canonical string rep and vise versa
        self.bin2canon = {eJ: 'e' + ''.join(str((eJ & 2**ei).bit_length()).replace('0', '') for ei in range(0, self.d))
                          for eJ in range(2 ** self.d)}
        self.bin2canon[0] = '1'
        self.canon2bin = dict(sorted({c: b for b, c in self.bin2canon.items()}.items(), key=lambda x: (len(x[0]), x[0])))

        self.swaps, self.signs, self.cayley = self._prepare_signs_and_cayley()

        # Make convenient shorthand constructors for commonly used vector types.
        for grade, name in enumerate(['scalar', 'vector', 'bivector', 'trivector', 'quadvector'][:min(self.d+1, 5)]):
            setattr(self, name, partial(self.purevector, grade=grade))
            setattr(self, 'pseudo' + name, partial(self.purevector, grade=self.d - grade))

        self.blades = {k: MultiVector.fromkeysvalues(self, keys=(v,), values=(1,))
                       for k, v in self.canon2bin.items()}
        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

        # Prepare OperatorDict's
        operators = (f for f in fields(self) if 'codegen' in f.metadata)
        for f in operators:
            setattr(self, f.name, OperatorDict(name=f.name, codegen=f.metadata['codegen'], algebra=self))

    def __len__(self):
        return 2 ** self.d

    @cached_property
    def indices_for_grade(self):
        """
        Mapping from the grades to the indices for that grade. E.g. in 2D VGA, this returns
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
                    sign *= self.signature[int(key) - 1]
                count[key] = value % 2
            signs[self.canon2bin[eI], self.canon2bin[eJ]] = sign

            # Make the Cayley table.
            if sign:
                prod = ''.join(key*value for key, value in count.items())
                sign = '-' if sign == -1 else ''
                cayley[eI, eJ] = f'{sign}{"e" if prod != "" else "1"}{prod}'
            else:
                cayley[eI, eJ] = f'0'
        return swaps_arr, signs, cayley

    def multivector(self, *args, **kwargs):
        """ Create a new :class:`~kingdon.algebra.MultiVector`. """
        return MultiVector(self, *args, **kwargs)

    def evenmv(self, *args, **kwargs):
        """ Create a new :class:`~kingdon.algebra.MultiVector` in the even subalgebra. """
        grades = tuple(filter(lambda x: x % 2 == 0, range(self.d + 1)))
        return MultiVector(self, *args, grades=grades, **kwargs)

    def oddmv(self, *args, **kwargs):
        """
        Create a new :class:`~kingdon.algebra.MultiVector` of odd grades.
        (There is technically no such thing as an odd subalgebra, but
        otherwise this is similar to :class:`~kingdon.algebra.Algebra.evenmv`.)
        """
        grades = tuple(filter(lambda x: x % 2 == 1, range(self.d + 1)))
        return MultiVector(self, *args, grades=grades, **kwargs)

    def purevector(self, *args, grade, **kwargs):
        """
        Create a new :class:`~kingdon.algebra.MultiVector` of a specific grade.

        :param grade: Grade of the mutivector to create.
        """
        return MultiVector(self, *args, grades=(grade,), **kwargs)
