import math
from itertools import combinations, product, repeat, chain
from functools import reduce, partial
from collections.abc import Sequence, Mapping
from collections import defaultdict, Counter
from dataclasses import dataclass, field, replace

import numpy as np
from sympy import Dummy, Symbol, symbols, Expr, simplify

from kingdon.codegen import codegen_gp, codegen_sp, codegen_cp, codegen_ip, codegen_op, codegen_rp


class AlgebraError(Exception):
    pass


@dataclass
class Algebra:
    p: int = 0
    q: int = 0
    r: int = 0
    d: int = field(init=False, repr=False)  # Total number of dimensions
    signature: list[int] = field(init=False, repr=False)

    # Dictionaries that cache previously symbolically optimized lambda functions between elements.
    _gp: dict = field(default_factory=dict, init=False, repr=False)  # geometric product dict
    _sp: dict = field(default_factory=dict, init=False, repr=False)  # conjugation dict
    _cp: dict = field(default_factory=dict, init=False, repr=False)  # commutator product dict
    _ip: dict = field(default_factory=dict, init=False, repr=False)  # inner product dict
    _op: dict = field(default_factory=dict, init=False, repr=False)  # exterior product dict
    _rp: dict = field(default_factory=dict, init=False, repr=False)  # regressive product dict

    # Mappings from binary to canonical reps. e.g. 0b01 <-> 'e1', 0b11 <-> 'e12'.
    canon2bin: dict = field(init=False, repr=False)
    bin2canon: dict = field(init=False, repr=False)

    # Options for the algebra
    cse: bool = field(default=True)  # Common Subexpression Elimination (CSE)
    precompute: str = field(default='none')  # Precompute (common) products. Options: 'none' (default), 'all', 'common'.
    numba: bool = field(default=False)  # Enable numba just-in-time compilation
    graded: bool = field(default=False)  # If true, precompute products per grade.

    signs: dict = field(init=False, repr=False)
    cayley: dict = field(init=False, repr=False)
    blades: dict = field(init=False, repr=False)
    pss: object = field(init=False, repr=False)

    def __post_init__(self):
        self.d = self.p + self.q + self.r
        self.signature = np.array([1]*self.p + [-1]*self.q + [0]*self.r)

        # Setup mapping from binary to canonical string rep and vise versa
        self.bin2canon = {eJ: 'e' + ''.join(str((eJ & 2**ei).bit_length()).replace('0', '') for ei in range(0, self.d))
                          for eJ in range(2 ** self.d)}
        self.bin2canon[0] = '1'
        self.canon2bin = dict(sorted({c: b for b, c in self.bin2canon.items()}.items(), key=lambda x: (len(x[0]), x[0])))

        self.signs, self.cayley = self._prepare_signs_and_cayley()

        # Make convenient shorthand constructors for commonly used vector types.
        for grade, name in enumerate(['scalar', 'vector', 'bivector', 'trivector', 'quadvector'][:min(self.d+1, 5)]):
            setattr(self, name, partial(self.purevector, grade=grade))
            setattr(self, 'pseudo' + name, partial(self.purevector, grade=self.d - grade))

        self.blades = {k: self.multivector({v: 1}) for k, v in self.canon2bin.items()}
        self.pss = self.blades[self.bin2canon[2 ** self.d - 1]]

    def _prepare_signs_and_cayley(self):
        """
        Prepares two dicts whose keys are two basis-blades in (binary rep) and the result is either
        just the sign (1, -1, 0) of the corresponding multiplication, or the full result.
        The full result is essentially the Cayley table, if printed as a table.

        E.g. in :math:`\mathbb{R}_2`, sings[(0b11, 0b11)] = -1.
        """
        cayley = {}
        signs = {}
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
        return signs, cayley

    def multivector(self, *args, **kwargs):
        return MultiVector(*args, algebra=self, **kwargs)

    def purevector(self, *args, grade, **kwargs):
        return PureVector(*args, grade=grade, algebra=self, **kwargs)


@dataclass
class MultiVector:
    algebra: Algebra = field(kw_only=True)
    vals: dict[int] = field(default_factory=lambda: defaultdict(int))
    name: str = field(default_factory=str)

    grades: list[int] = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.vals, Mapping):
            try:
                self.vals = {key if key in self.algebra.bin2canon else self.algebra.canon2bin[key]: val
                             for key, val in self.vals.items()}
            except KeyError:
                raise KeyError(f"Invalid key(s) in `vals`: keys should be `int` or canonical strings (e.g. `e12`)")
        elif len(self.vals) == len(self):
            self.vals = {i: val for i, val in enumerate(self.vals)}
        else:
            raise TypeError(f'`vals` should have length {len(self)}.')

        if self.name and not self.vals:
            # self.vals was in fact empy, but we do have a name. So we are in symbolic mode.
            self.vals = dict(zip(range(len(self)), symbols(' '.join(f'{self.name}{i}' for i in range(len(self))))))

        #TODO: extract grades present in self.vals
        self.grades = []

    def __len__(self):
        return 2 ** self.algebra.d

    def grade(self, grade):
        # TODO: support multiple grade selection.
        return replace(self, vals={k: v for k, v in self.vals.items() if self._bin_grade(k) == grade})

    def __neg__(self):
        return replace(self, vals={k: -v for k, v in self.vals.items()})

    def __invert__(self):
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
        return MultiVector(vals=vals, algebra=self.algebra)

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

    @staticmethod
    def _bin_grade(value):
        """ Retrieves the grade of a binary index. """
        return bin(value).count('1')

    def asmatrix(self):
        raise NotImplementedError

    def _multiplication(self, other, func_dictionary, codegen):
        """ Helper function for all multiplication types such as gp, sp, cp etc. """
        if self.algebra != other.algebra:
            raise AlgebraError("Cannot multiply elements of different algebra's.")

        key = (tuple(self.vals), tuple(other.vals))
        if key not in func_dictionary:
            x = replace(self, vals={ek: Symbol(f'a{ek}') for ek in self.vals})
            y = replace(other, vals={ek: Symbol(f'b{ek}') for ek in other.vals})
            keys, func = func_dictionary[key] = codegen(x, y)
        else:
            keys, func = func_dictionary[key]

        args = chain(self.vals.values(), other.vals.values())
        # TODO: investigate the use of np.any, because this might break ducktyping.
        res_vals = defaultdict(int, {k: v for k, v in zip(keys, func(*args))
                                     if (np.any(v) if not isinstance(v, Expr) else simplify(v))})

        return self.algebra.multivector(vals=res_vals)

    def gp(self, other):
        if not hasattr(other, 'algebra'):
            # Assume scalar multiplication
            return replace(self, vals={k: v * other for k, v in self.vals.items()})

        return self._multiplication(other, func_dictionary=self.algebra._gp, codegen=codegen_gp)

    __mul__ = __rmul__ = gp

    def sp(self, other):
        """ Apply `x := self` to `y := other` under conjugation: `x*y*~x`. """
        return self._multiplication(other, func_dictionary=self.algebra._sp, codegen=codegen_sp)

    __rshift__ = sp

    def cp(self, other):
        """ Calculate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`. """
        return self._multiplication(other, func_dictionary=self.algebra._cp, codegen=codegen_cp)
    # __rshift__ = sp

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

@dataclass
class PureVector(MultiVector):
    """ A PureVector is a MultiVector of a single grade. """
    grade: int = -1

    def __post_init__(self):
        if not 0 <= self.grade <= self.algebra.d:
            raise ValueError(f'`grade` needs to be a value between 0 and {self.algebra.d}.')
        # Make #self.vals into a valid mapping if it wasn't already.
        if not isinstance(self.vals, Mapping):
            indices = [self.algebra.canon2bin['e' + ''.join(str(i) for i in comb)]
                       for comb in combinations(range(1, self.algebra.d + 1), r=self.grade)]
            self.vals = {k: v for k, v in zip(indices, self.vals)}

        if self.name and not self.vals:
            # self.vals was in fact empy, but we do a name. So we are in symbolic mode.
            self.vals = {self.algebra.canon2bin['e' + ''.join(str(i) for i in comb)]:
                            Symbol(self.name + ''.join(str(i) for i in comb))
                         for comb in combinations(range(1, self.algebra.d + 1), r=self.grade)}

        super().__post_init__()
        if not all(self.grade == bin(i).count("1") for i in self.vals):
            raise ValueError(f"All keys in `vals` should be of grade r={self.grade}.")

    def __len__(self):
        return math.comb(self.algebra.d, self.grade)

class Versor: pass
