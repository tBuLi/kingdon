from itertools import combinations, product, chain, groupby
from functools import partial, cached_property, reduce
from collections.abc import Mapping
from collections import defaultdict, Counter
from dataclasses import dataclass, field, replace, fields
from contextlib import contextmanager

import numpy as np
from sympy import Symbol, Expr, simplify

from kingdon.codegen import (
    codegen_gp, codegen_conj, codegen_cp, codegen_ip, codegen_op,
    codegen_rp, codegen_acp, codegen_proj, codegen_sp, codegen_lc, codegen_inv, codegen_rc, _lambdify_mv
)
from kingdon.matrixreps import matrix_rep
# from kingdon.module_builder import predefined_modules

operation_field = partial(field, default_factory=dict, init=False, repr=False, compare=False)


class AlgebraError(Exception):
    pass


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

    # Dictionaries that cache previously symbolically optimized lambda functions between elements.
    _gp: dict = operation_field(metadata={'codegen': codegen_gp, 'syntax': '__mul__'})  # geometric product dict
    _conj: dict = operation_field(metadata={'codegen': codegen_conj})  # conjugation dict
    _cp: dict = operation_field(metadata={'codegen': codegen_cp})  # commutator product dict
    _acp: dict = operation_field(metadata={'codegen': codegen_acp})  # anti-commutator product dict
    _ip: dict = operation_field(metadata={'codegen': codegen_ip})  # inner product dict
    _sp: dict = operation_field(metadata={'codegen': codegen_sp})  # Scalar product dict
    _lc: dict = operation_field(metadata={'codegen': codegen_lc})  # left-contraction
    _rc: dict = operation_field(metadata={'codegen': codegen_rc})  # right-contraction
    _op: dict = operation_field(metadata={'codegen': codegen_op})  # exterior product dict
    _rp: dict = operation_field(metadata={'codegen': codegen_rp})  # regressive product dict
    _proj: dict = operation_field(metadata={'codegen': codegen_proj})  # projection dict
    _inv: dict = operation_field(metadata={'codegen': codegen_inv})  # projection dict

    # Mappings from binary to canonical reps. e.g. 0b01 = 1 <-> 'e1', 0b11 = 3 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False, compare=False)
    bin2canon: dict = field(init=False, repr=False, compare=False)

    # Options for the algebra
    cse: bool = field(default=True)  # Common Subexpression Elimination (CSE)
    precompute: str = field(default='none')  # Precompute (common) products. Options: 'none' (default), 'all', 'common'.
    numba: bool = field(default=False)  # Enable numba just-in-time compilation
    graded: bool = field(default=False)  # If true, precompute products per grade.

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

        self.blades = {k: self.multivector({v: 1}) for k, v in self.canon2bin.items()}
        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

    def __len__(self):
        return 2 ** self.d

    @cached_property
    def indices_for_grade(self):
        """
        Mapping from the grades to the indices for that grade. E.g. in 2D VGA, this returns
        {0: (0,), 1: (1, 2), 2: (3,)}
        """
        key = lambda i: bin(i).count('1')
        return {grade: tuple(inds)
                for grade, inds in groupby(sorted(range(len(self)), key=key), key=key)}

    @cached_property
    def indices_for_grades(self):
        """
        Mapping from a sequence of grades to the corresponding indices. E.g. in 2D VGA, this returns
        {(0,): (0,), (1,): (1, 2), (2,): (3,), (0, 1): (0, 1, 2),
         (0, 2): (0, 3), (1, 2): (1, 2, 3), (0, 1, 2): (0, 1, 2, 3)}
        """
        all_grade_combs = chain(*(combinations(range(0, self.d + 1), r=j) for j in range(1, len(self) + 1)))
        return {comb: sum((self.indices_for_grade[grade] for grade in comb), ())
                for comb in all_grade_combs}

    @cached_property
    def matrix_basis(self):
        return matrix_rep(self.p, self.q, self.r)

    @cached_property
    def _binary_operations(self):
        return {f.name: dict(f.metadata) for f in fields(self) if f.metadata}

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

    def mvfromtrusted(self, *args, **kwargs):
        return MultiVector.fromtrusted(*args, algebra=self, **kwargs)

    def multivector(self, *args, **kwargs):
        """ Create a new :class:`~kingdon.algebra.MultiVector`. """
        return MultiVector(self, *args, **kwargs)

    def evenmv(self, *args, **kwargs):
        """ Create a new :class:`~kingdon.algebra.MultiVector` in the even subalgebra. """
        grades = tuple(filter(lambda x: x % 2 == 0, range(self.d + 1)))
        return MultiVector.withgrades(self, *args, grades=grades, **kwargs)

    def oddmv(self, *args, **kwargs):
        """
        Create a new :class:`~kingdon.algebra.MultiVector` of odd grades.
        (There is technically no such thing as an odd subalgebra, but
        otherwise this is similar to :class:`~kingdon.algebra.Algebra.evenmv`.)
        """
        grades = tuple(filter(lambda x: x % 2 == 1, range(self.d + 1)))
        return MultiVector.withgrades(self, *args, grades=grades, **kwargs)

    def purevector(self, *args, grade, **kwargs):
        """
        Create a new :class:`~kingdon.algebra.MultiVector` of a specific grade.

        :param grade: Grade of the mutivector to create.
        """
        return MultiVector.withgrades(self, *args, grades=(grade,), **kwargs)


@dataclass
class MultiVector:
    algebra: Algebra = field()
    vals: dict = field(default_factory=lambda: defaultdict(int))
    name: str = field(default_factory=str)

    def __post_init__(self):
        if isinstance(self.vals, Mapping):
            try:
                self.vals = {key if key in self.algebra.bin2canon else self.algebra.canon2bin[key]:
                             val if not isinstance(val, str) else Symbol(val)
                             for key, val in self.vals.items()}
            except KeyError:
                raise KeyError(f"Invalid key(s) in `vals`: keys should be `int` or canonical strings (e.g. `e12`)")
        elif len(self.vals) == len(self.algebra):
            self.vals = {i: val for i, val in enumerate(self.vals)}
        else:
            raise TypeError(f'`vals` should be a mapping, or be a sequence of length {len(self.algebra)}.')

        if self.name and not self.vals:
            # self.vals was in fact empy, but we do have a name. So we are in symbolic mode.
            self.vals = {k: Symbol(f'{self.name}{self.algebra.bin2canon[k][1:]}') for k in range(len(self.algebra))}

    @classmethod
    def withgrades(cls, algebra, vals=None, name=None, *, grades, **kwargs):
        """
        Alternative constructor which creates a MultiVector instance with only the specified grades.

        Example usage::

            >>> alg = Algebra(2)
            >>> u = MultiVector.withgrades(alg, [1.2, 3.4], grades=1)
            >>> u
            1.2e1 + 3.4e2

        """
        grades = grades if isinstance(grades, tuple) else tuple(grades)

        if not all(0 <= grade <= algebra.d for grade in grades):
            raise ValueError(f'Each grade in `grades` needs to be a value between 0 and {algebra.d}.')

        if name and vals is None:
            # vals was in fact empy, but we do have a name. So we are in symbolic mode.
            vals = {k: Symbol(f'{name}{algebra.bin2canon[k][1:]}') for k in algebra.indices_for_grades[grades]}

        if not isinstance(vals, Mapping):
            # If vals is a sequence, it should be of the right length so it can be turned into a mapping.
            if len(vals) == len(algebra.indices_for_grades[grades]):
                indices = algebra.indices_for_grades[grades]
                vals = {k: v for k, v in zip(indices, vals)}
            else:
                raise TypeError(f'`vals` should be a mapping, or be a sequence of length {len(algebra)}.')
        elif isinstance(vals, Mapping):
            try:
                vals = {key if key in algebra.bin2canon else algebra.canon2bin[key]: val
                             for key, val in vals.items()}
            except KeyError:
                raise KeyError(f"Invalid key(s) in `vals`: keys should be `int` or canonical strings (e.g. `e12`)")
            if not set(vals.keys()) <= set(algebra.indices_for_grades[grades]):
                raise ValueError(f"All keys in `vals` should be of grades {grades}.")

        return cls(algebra=algebra, vals=vals, **kwargs)

    @classmethod
    def fromtrusted(cls, algebra, vals):
        """
        Create a new MultiVector without performing any sanity checking on the input.
        This is meant for internal use only, as the lack off sanity checking increases performance.
        """
        obj = cls.__new__(cls, algebra=algebra)
        obj.algebra = algebra
        obj.vals = vals
        obj.name = ''
        return obj

    @classmethod
    def frommatrix(cls, algebra, matrix):
        """
        Initiate a multivector from a matrix. This matrix is assumed to be
        generated by :class:`~kingdon.algebra.MultiVector.asmatrix`, and
        thus we only read the first column of the input matrix.
        """
        obj = cls(algebra=algebra, vals=matrix[:, 0])
        return obj

    def __len__(self):
        return len(self.vals)

    @cached_property
    def grades(self):
        """ Tuple of the grades present in `self`. """
        return tuple(sorted({bin(ind).count('1') for ind in self.vals}))

    def grade(self, grades):
        """
        Returns a new  :class:`~kingdon.algebra.MultiVector` instance with
        only the selected `grades` from `self`.

        :param grades: tuple or int, grades to select.
        """
        if isinstance(grades, int):
            grades = (grades,)
        elif not isinstance(grades, tuple):
            grades = tuple(grades)

        vals = {k: self.vals[k]
                for k in self.algebra.indices_for_grades[grades] if k in self.vals}
        return self.algebra.multivector(vals)

    @cached_property
    def issymbolic(self):
        """ True if this mv contains Symbols, False otherwise. """
        return any(isinstance(v, Expr) for v in self.vals.values())

    def __neg__(self):
        return replace(self, vals={k: -v for k, v in self.vals.items()})

    def __invert__(self):  # reversion
        return replace(self, vals={k: (-1)**(bin(k).count("1") // 2) * v for k, v in self.vals.items()})

    def normsq(self):
        return self * ~self

    def inv(self):
        """ Inverse of this multivector. """
        return self._unary_operation(func_dictionary=self.algebra._inv, codegen=codegen_inv)

    def __add__(self, other):
        if not isinstance(other, MultiVector):
            other = self.algebra.multivector({0: other})
        vals = self.vals.copy()
        for k, v in other.vals.items():
            if k in vals:
                vals[k] += v
            else:
                vals[k] = v
        # TODO: check grades and produce the corresponding type. MV is usually to general.
        return self.algebra.mvfromtrusted(vals=vals)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        if not hasattr(other, 'algebra'):
            # Assume scalar multiplication
            return replace(self, vals={k: v / other for k, v in self.vals.items()})
        elif 0 in other.vals and len(other.vals) == 1:
            # other is essentially a scalar.
            return replace(self, vals={k: v / other[0] for k, v in self.vals.items()})
        elif not other.vals:
            raise ZeroDivisionError
        raise NotImplementedError

    def __str__(self):
        if isinstance(self.vals, Mapping) and self.vals:
            canon_sorted_vals = sorted(self.vals.items(), key=lambda x: (len(self.algebra.bin2canon[x[0]]), self.algebra.bin2canon[x[0]]))
            return ' + '.join([f'({val}) * {self.algebra.bin2canon[key]}' for key, val in canon_sorted_vals])
        else:
            return '0'
    
    __repr__ = __str__
    
    def __getitem__(self, item):
        return self.vals.get(item if item in self.algebra.bin2canon else self.algebra.canon2bin[item], 0)

    def __setitem__(self, item, value):
        self.vals[item if item in self.algebra.bin2canon else self.algebra.canon2bin[item]] = value

    @cached_property
    def free_symbols(self):
        return reduce(lambda tot, x: tot | x, (v.free_symbols for v in self.vals.values()))

    @cached_property
    def _callable(self):
        """ Return the callable function for this MV. """
        return _lambdify_mv(sorted(self.free_symbols, key=lambda x: x.name), self)

    def __call__(self, *args, **kwargs):
        if not self.free_symbols:
            return self
        keys_out, func = self._callable
        res_vals = {k: v for k, v in zip(keys_out, func(*args, **kwargs))}
        return self.algebra.mvfromtrusted(vals=res_vals)

    def asmatrix(self):
        """ Returns a matrix representation of this multivector. """
        return sum(v * self.algebra.matrix_basis[k] for k, v in self.vals.items())

    def _unary_operation(self, func_dictionary, codegen):
        """ Helper function for all unary operations such as inv, dual, pow etc. """
        keys_in = tuple(self.vals)
        if keys_in not in func_dictionary:
            x = self.algebra.multivector(vals={ek: Symbol(f'a{self.algebra.bin2canon[ek][1:]}')
                                               for ek in keys_in})
            keys_out, func = func_dictionary[keys_in] = codegen(x)
        else:
            keys_out, func = func_dictionary[keys_in]

        args = tuple(self.vals.values())
        if self.issymbolic:
            res_vals = {k: v for k, v in zip(keys_out, func(args))
                        if (True if not isinstance(v, Expr) else simplify(v))}
        else:
            res_vals = {k: v for k, v in zip(keys_out, func(args))}

        return self.algebra.mvfromtrusted(vals=res_vals)

    def _binary_operation(self, other, func_dictionary, codegen):
        """ Helper function for all multiplication types such as gp, sp, cp etc. """
        if not isinstance(other, MultiVector):
            # Assume scalar multiplication, turn into a mv.
            other = self.algebra.scalar(vals={0: other})
        elif self.algebra != other.algebra:
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = (tuple(self.vals), tuple(other.vals))
        if keys_in not in func_dictionary:
            x = self.algebra.multivector(vals={ek: Symbol(f'a{self.algebra.bin2canon[ek][1:]}')
                                               for ek in keys_in[0]})
            y = self.algebra.multivector(vals={ek: Symbol(f'b{self.algebra.bin2canon[ek][1:]}')
                                               for ek in keys_in[1]})
            keys_out, func = func_dictionary[keys_in] = codegen(x, y)
        else:
            keys_out, func = func_dictionary[keys_in]

        args = tuple(self.vals.values()), tuple(other.vals.values())
        if self.issymbolic or other.issymbolic:
            res_vals = {k: v for k, v in zip(keys_out, func(*args))
                        if (True if not isinstance(v, Expr) else simplify(v))}
        else:
            res_vals = {k: v for k, v in zip(keys_out, func(*args))}

        return self.algebra.mvfromtrusted(vals=res_vals)

    def gp(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._gp, codegen=codegen_gp)

    __mul__ = __rmul__ = gp

    def conj(self, other):
        """ Apply `x := self` to `y := other` under conjugation: `x*y*~x`. """
        return self._binary_operation(other, func_dictionary=self.algebra._conj, codegen=codegen_conj)

    def proj(self, other):
        """
        Project :code:`x := self` onto :code:`y := other`: :code:`x @ y = (x | y) * ~y`.
        For correct behavior, :code:`x` and :code:`y` should be normalized (k-reflections).
        """
        return self._binary_operation(other, func_dictionary=self.algebra._proj, codegen=codegen_proj)

    __matmul__ = proj

    def cp(self, other):
        """
        Calculate the commutator product of :code:`x := self` and :code:`y := other`:
        :code:`x.cp(y) = 0.5*(x*y-y*x)`.
        """
        return self._binary_operation(other, func_dictionary=self.algebra._cp, codegen=codegen_cp)

    def ip(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._ip, codegen=codegen_ip)

    __or__ = ip

    def op(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._op, codegen=codegen_op)

    __xor__ = op

    def lc(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._lc, codegen=codegen_lc)

    __lshift__ = lc

    def rc(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._rc, codegen=codegen_rc)

    __rshift__ = rc

    def sp(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._sp, codegen=codegen_sp)

    def rp(self, other):
        return self._binary_operation(other, func_dictionary=self.algebra._rp, codegen=codegen_rp)

    __and__ = rp

    def dual(self, kind='auto'):
        """
        Compute the dual of `self`. There are three different kinds of duality in common usage.
        The first is polarity, which is simply multiplying by the inverse PSS. This is the only game in town for
        non-degenerate metrics (Algebra.r = 0). However, for degenerate spaces this no longer works, and we have
        two popular options: Poincaré and Hodge duality.

        By default, `kingdon` will use polarity in non-degenerate spaces, and Hodge duality for spaces with
        `Algebra.r = 1`. For spaces with `r > 2`, little to no literature exists, and you are on your own.

        :param kind: if 'auto' (default), :mod:`kingdon` will try to determine the best dual on the
            basis of the signature of the space. See explenation above.
            To ensure polarity, use :code:`kind='polarity'`, and to ensure Hodge duality,
            use :code:`kind='hodge'`.
        """
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self * self.algebra.pss.inv()
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.algebra.multivector(
                vals={2**self.algebra.d - 1 - eI: self.algebra.signs[eI, 2**self.algebra.d - 1 - eI] * val
                      for eI, val in self.vals.items()}
            )
        elif kind == 'auto':
            raise Exception('Cannot select a suitable dual in auto mode for this algebra.')
        else:
            raise ValueError(f'No dual found for kind={kind}.')

    def undual(self, kind='auto'):
        """
        Compute the undual of `self`. See :class:`~kingdon.algebra.MultiVector.dual` for more information.
        """
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self * self.algebra.pss
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.algebra.multivector(
                vals={2**self.algebra.d - 1 - eI: self.algebra.signs[2**self.algebra.d - 1 - eI, eI] * val
                      for eI, val in self.vals.items()}
            )
        elif kind == 'auto':
            raise Exception('Cannot select a suitable undual in auto mode for this algebra.')
        else:
            raise ValueError(f'No undual found for kind={kind}.')


class GradedMultiplication:
    def _binary_operation(self, other, func_dictionary, codegen):
        """ Helper function for all multiplication types such as gp, sp, cp etc. """
        if self.algebra != other.algebra:
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = (self.algebra.indices_for_grades[self.grades],
                   self.algebra.indices_for_grades[other.grades])
        if keys_in not in func_dictionary:
            x = self.algebra.multivector(vals={ek: Symbol(f'a{self.algebra.bin2canon[ek][1:]}')
                                               for ek in keys_in[0]})
            y = self.algebra.multivector(vals={ek: Symbol(f'b{self.algebra.bin2canon[ek][1:]}')
                                               for ek in keys_in[1]})
            keys_out, func = func_dictionary[keys_in] = codegen(x, y)
        else:
            keys_out, func = func_dictionary[keys_in]

        args = chain((self.vals.get(i, 0) for i in keys_in[0]),
                     (other.vals.get(i, 0) for i in keys_in[1]))
        res_vals = defaultdict(int, {k: v for k, v in zip(keys_out, func(*args))
                                     if (True if v.__class__ is not Expr else simplify(v))})

        return self.algebra.mvfromtrusted(vals=res_vals)
