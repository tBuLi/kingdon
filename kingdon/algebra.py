from itertools import combinations, product, chain, groupby
from functools import partial, cached_property
from collections.abc import Mapping
from collections import defaultdict, Counter
from dataclasses import dataclass, field, replace

import numpy as np
from sympy import Symbol, Expr, simplify

from kingdon.codegen import codegen_gp, codegen_sp, codegen_cp, codegen_ip, codegen_op, codegen_rp


class AlgebraError(Exception):
    pass


@dataclass
class Algebra:
    p: int = 0
    q: int = 0
    r: int = 0
    d: int = field(init=False, repr=False, compare=False)  # Total number of dimensions
    signature: list[int] = field(init=False, repr=False, compare=False)

    # Dictionaries that cache previously symbolically optimized lambda functions between elements.
    _gp: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # geometric product dict
    _sp: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # conjugation dict
    _cp: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # commutator product dict
    _ip: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # inner product dict
    _op: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # exterior product dict
    _rp: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # regressive product dict

    # Mappings from binary to canonical reps. e.g. 0b01 <-> 'e1', 0b11 <-> 'e12'.
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

    def _prepare_signs_and_cayley(self):
        """
        Prepares two dicts whose keys are two basis-blades in (binary rep) and the result is either
        just the sign (1, -1, 0) of the corresponding multiplication, or the full result.
        The full result is essentially the Cayley table, if printed as a table.

        E.g. in :math:`\mathbb{R}_2`, sings[(0b11, 0b11)] = -1.
        """
        cayley = {}
        sign_dict = {}
        swap_dict = {}
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
            swap_dict[self.canon2bin[eI], self.canon2bin[eJ]] = swaps

            # Remove even powers of basis-vectors.
            sign = -1 if swaps % 2 else 1
            count = Counter(prod)
            for key, value in count.items():
                if value // 2:
                    sign *= self.signature[int(key) - 1]
                count[key] = value % 2
            sign_dict[self.canon2bin[eI], self.canon2bin[eJ]] = sign

            # Make the Cayley table.
            if sign:
                prod = ''.join(key*value for key, value in count.items())
                sign = '-' if sign == -1 else ''
                cayley[eI, eJ] = f'{sign}{"e" if prod != "" else "1"}{prod}'
            else:
                cayley[eI, eJ] = f'0'
        return swap_dict, sign_dict, cayley

    def mvfromtrusted(self, *args, **kwargs):
        return MultiVector.fromtrusted(*args, algebra=self, **kwargs)

    def multivector(self, *args, **kwargs):
        return MultiVector(*args, algebra=self, **kwargs)

    def purevector(self, *args, grade, **kwargs):
        return MultiVector.withgrades(self, *args, grades=(grade,), **kwargs)


@dataclass
class MultiVector:
    algebra: Algebra = field(kw_only=True)
    vals: dict[int] = field(default_factory=lambda: defaultdict(int))
    name: str = field(default_factory=str)

    def __post_init__(self):
        if isinstance(self.vals, Mapping):
            try:
                self.vals = {key if key in self.algebra.bin2canon else self.algebra.canon2bin[key]: val
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

        Example usage:
        ..code:
            >>> alg = Algebra(2)
            >>> u = MultiVector.withgrades(alg, [1.2, 3.4], grades=(1,))
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
        elif not set(vals.keys()) <= set(algebra.indices_for_grades[grades]):
            raise ValueError(f"All keys in `vals` should be of grades {grades}.")

        return cls(algebra=algebra, vals=vals, **kwargs)

    @classmethod
    def fromtrusted(cls, algebra, vals):
        """
        Create a new MultiVector without performing any sanity checking on the input.
        This is meant for internal use only, as the lack off sanity checking increases performance.
        """
        obj = cls.__new__(cls)
        obj.algebra = algebra
        obj.vals = vals
        obj.name = ''
        return obj

    def __len__(self):
        return len(self.vals)

    @cached_property
    def grades(self):
        """ Determines the grades present in `self`. """
        return tuple(sorted({bin(ind).count('1') for ind in self.vals}))

    def grade(self, grades):
        """ Returns a new MultiVector instance with only the selected `grades` from `self`. """
        if isinstance(grades, int):
            grades = (grades,)
        elif not isinstance(grades, tuple):
            grades = tuple(grades)

        vals = {k: self.vals[k]
                for k in self.algebra.indices_for_grades[grades] if k in self.vals}
        return self.algebra.multivector(vals)

    def __neg__(self):
        return replace(self, vals={k: -v for k, v in self.vals.items()})

    def __invert__(self):  # reversion
        return replace(self, vals={k: (-1)**(bin(k).count("1") // 2) * v for k, v in self.vals.items()})

    def normsq(self):
        return self * ~self

    def inv(self):
        """ Inverse of this multivector. """
        return self / self.normsq()

    def __add__(self, other):
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
        raise NotImplementedError

    __rtruediv__ = __truediv__

    def __str__(self):
        if isinstance(self.vals, Mapping):
            return ' + '.join([f'({val}) * {self.algebra.bin2canon[key]}' for key, val in self.vals.items()])
        return ' + '.join([f'({val}) * {self.algebra.bin2canon[i]}' for i, val in enumerate(self.vals)])

    def __getitem__(self, item):
        return self.vals[item if item in self.algebra.bin2canon else self.algebra.canon2bin[item]]

    def __setitem__(self, item, value):
        self.vals[item if item in self.algebra.bin2canon else self.algebra.canon2bin[item]] = value

    def __call__(self):
        raise NotImplementedError

    def asmatrix(self):
        raise NotImplementedError

    def _multiplication(self, other, func_dictionary, codegen):
        """ Helper function for all multiplication types such as gp, sp, cp etc. """
        if self.algebra != other.algebra:
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = (tuple(self.vals), tuple(other.vals))
        if keys_in not in func_dictionary:
            # TODO: fix names
            x = self.algebra.multivector(vals={ek: Symbol(f'a{ek}') for ek in keys_in[0]})
            y = self.algebra.multivector(vals={ek: Symbol(f'b{ek}') for ek in keys_in[1]})
            keys_out, func = func_dictionary[keys_in] = codegen(x, y)
        else:
            keys_out, func = func_dictionary[keys_in]

        args = chain(self.vals.values(), other.vals.values())
        res_vals = {k: v for k, v in zip(keys_out, func(*args))
                    if (True if not isinstance(v, Expr) else simplify(v))}

        return self.algebra.mvfromtrusted(vals=res_vals)

    def gp(self, other):
        if not hasattr(other, 'algebra'):
            # Assume scalar multiplication, turn into a mv.
            other = self.algebra.scalar(vals={0: other})

        return self._multiplication(other, func_dictionary=self.algebra._gp, codegen=codegen_gp)

    __mul__ = __rmul__ = gp

    def sp(self, other):
        """ Apply `x := self` to `y := other` under conjugation: `x*y*~x`. """
        return self._multiplication(other, func_dictionary=self.algebra._sp, codegen=codegen_sp)

    __rshift__ = sp

    def cp(self, other):
        """ Calculate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`. """
        return self._multiplication(other, func_dictionary=self.algebra._cp, codegen=codegen_cp)

    def ip(self, other):
        return self._multiplication(other, func_dictionary=self.algebra._ip, codegen=codegen_ip)

    __or__ = ip

    def op(self, other):
        return self._multiplication(other, func_dictionary=self.algebra._op, codegen=codegen_op)

    __xor__ = op

    def rp(self, other):
        return self._multiplication(other, func_dictionary=self.algebra._rp, codegen=codegen_rp)

    __and__ = rp

    def dual(self, dual='auto'):
        """
        Compute the dual of `self`. There are three different kinds of duality in common usage.
        The first is polarity, which is simply multiplying by the inverse PSS. This is the only game in town for
        non-degenerate metrics (Algebra.r = 0). However, for degenerate spaces this no longer works, and we have
        two popular options: Poincaré and Hodge duality.

        By default, `kingdon` will use polarity in non-degenerate spaces, and Hodge duality for spaces with
        `Algebra.r = 1`. For spaces with `r > 2`, little to no literature exists, and you are on your own.
        """
        if dual == 'polarity' or dual == 'auto' and self.algebra.r == 0:
            return self * (1 / self.algebra.pss)
        else:
            raise NotImplementedError


class GradedMultiplication:
    def _multiplication(self, other, func_dictionary, codegen):
        """ Helper function for all multiplication types such as gp, sp, cp etc. """
        if self.algebra != other.algebra:
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        keys_in = (self.algebra.indices_for_grades[self.grades],
                   self.algebra.indices_for_grades[other.grades])
        if keys_in not in func_dictionary:
            x = self.algebra.multivector(vals={ek: Symbol(f'a{ek}') for ek in keys_in[0]})
            y = self.algebra.multivector(vals={ek: Symbol(f'b{ek}') for ek in keys_in[1]})
            keys_out, func = func_dictionary[keys_in] = codegen(x, y)
        else:
            keys_out, func = func_dictionary[keys_in]

        args = chain((self.vals.get(i, 0) for i in keys_in[0]),
                     (other.vals.get(i, 0) for i in keys_in[1]))
        res_vals = defaultdict(int, {k: v for k, v in zip(keys_out, func(*args))
                                     if (True if v.__class__ is not Expr else simplify(v))})

        return self.algebra.mvfromtrusted(vals=res_vals)
