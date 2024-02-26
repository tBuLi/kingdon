import itertools
from dataclasses import dataclass, field
from typing import List

from sympy import Mul, Add, Symbol, RealNumber

from kingdon.codegen import power_supply

def compare(a, b):
    if a is None: return 1
    if b is None: return -1

    la = len(a)
    lb = len(b)
    l = min(la, lb)
    for i in range(1, l):
        if a[i] < b[i]: return -1
        elif a[i] > b[i]: return 1
    return la - lb


@dataclass
class Polynomial:
    args: List[list] = field(init=False)

    def __init__(self, coeff):
        if isinstance(coeff, self.__class__):
            self.args = coeff.args
        elif isinstance(coeff, (list, tuple)):
            self.args = coeff
        elif isinstance(coeff, (int, float)):
            self.args = [[coeff]]
        elif isinstance(coeff, str):
            self.args = [[1, coeff]] if coeff[0] != "-" else [[-1, coeff[1:]]]

    @classmethod
    def fromname(cls, name):
        return cls([[1, name]])

    def __len__(self):
        return len(self.args)

    def __getitem__(self, item):
        return self.args[item]

    def __eq__(self, other):
        if other == 0 and (not self.args or self.args == [[0]]): return True
        if other == 1 and self.args == [[1]]: return True
        if self.__class__ != other.__class__: return False
        return self.args == other.args

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        ai = bi = 0
        al = len(self)
        bl = len(other)
        res = []

        while not (ai == al and bi == bl):
            ea = self[ai] if ai < al else None
            eb = other[bi] if bi < bl else None
            diff = compare(ea, eb)
            if diff < 0:
                res.append(ea)
                ai += 1
            elif diff > 0:
                res.append(eb)
                bi += 1
            else:
                ea = ea.copy()
                ea[0] += eb[0]
                if ea[0] != 0:
                    res.append(ea)
                ai += 1
                bi += 1
        return self.__class__(res)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if self == 0 or other == 0:
            return self.__class__([])

        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        res = Polynomial([])
        al = len(self)
        bl = len(other)
        for ai, bi in itertools.product(range(0, al), range(0, bl)):
            A = self[ai]
            B = other[bi]
            C = [A[0] * B[0]]
            i = 1
            j = 1
            while i < len(A) or j < len(B):
                ea = A[i] if i < len(A) else None
                eb = B[j] if j < len(B) else None
                # if ea is None and eb is None: break
                if eb is None or (ea is not None and ea < eb):
                    if isinstance(ea, str): C.append(ea)
                    else: C[0] *= ea
                    i += 1
                else:
                    if isinstance(eb, str): C.append(eb)
                    else: C[0] *= eb
                    j += 1
            res = res + Polynomial([C])
        return Polynomial(res)

    __rmul__ = __mul__

    def __neg__(self):
        return self.__class__([[-monomial[0], *monomial[1:]] for monomial in self.args])

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power, modulo=None):
        *_, last = power_supply(self, power)
        return last

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return RationalPolynomial(self, other)
        # Assume scalar
        return self * (1 / other)

    def __str__(self):
        preprocessed = (monomial if len(monomial) == 1 else monomial[1:] if monomial[0] == 1 else monomial
                        for monomial in self.args)
        return " + ".join("*".join(str(x) for x in monomial if x != 1) if len(monomial) > 1 else str(monomial[0])
                          for monomial in preprocessed)

    def tosympy(self):
        """ Return a sympy version of this Polynomial. """
        preprocessed = (monomial if len(monomial) == 1 else monomial[1:] if monomial[0] == 1 else monomial
                        for monomial in self.args)
        sympified = ([Symbol(s) if s.__class__ == str else s for s in monomial]
                     for monomial in preprocessed)
        terms = (Mul(*monomial, evaluate=True) for monomial in sympified)
        res = Add(*terms, evaluate=True)
        return res

    def __bool__(self):
        if len(self.args) == 1:
            return bool(self.args[0][0])
        return bool(self.args)


@dataclass
class RationalPolynomial:
    numer: Polynomial = field(init=False)
    denom: Polynomial = field(init=False)

    def __init__(self, numer, denom=None):
        if isinstance(numer, self.__class__):
            numer = numer.numer
            denom = numer.denom
        elif isinstance(numer, (list, tuple)):
            numer = Polynomial(numer)
        if denom is None:
            denom = Polynomial([[1]])
        elif isinstance(denom, (list, tuple)):
            denom = Polynomial(denom)
        self.numer = numer
        self.denom = denom

        # elif isinstance(coeff, Polynomial):
        #     self.args = [coeff, Polynomial([[1]])]
        # elif isinstance(coeff, (list, tuple)):
        #     self.args = [Polynomial(coeff), Polynomial([[1]])]
        # else:
        #     raise NotImplementedError

    @classmethod
    def fromname(cls, name):
        return cls([[1, name]])

    def __eq__(self, other):
        if other == 0 and (self.numer == 0): return True
        if other == 1 and (self.numer == 1 and self.denom == 1): return True
        if self.__class__ != other.__class__: return False
        return self.numer == other.numer and self.denom == other.denom

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        if other == 0: return self
        if self == 0: return other

        na, da = self.numer, self.denom
        nb, db = other.numer, other.denom

        if len(da) == len(db) and da == db:
            nn = na + nb
            nd = da
        else:
            nn, nd = na * db + nb * da, da * db

        if nn == 0: return RationalPolynomial([])
        if len(nn) == len(nd) and nn == nd: return RationalPolynomial([[1]])
        return RationalPolynomial(nn, nd)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__([[other]])

        if self == 0: return self
        if other == 0: return other
        if other == 1: return self
        if self == 1: return other

        na, da = self.numer, self.denom
        nb, db = other.numer, other.denom
        numer, denom = na * nb, da * db

        if numer == 0: return RationalPolynomial([[0]])
        if len(numer) == len(denom) and numer == denom: return RationalPolynomial([[1]])
        if len(numer) == 1 and len(denom) == 1:
            # Remove common factors from simple expressions
            fl1, fl2 = numer[0], denom[0]
            nnn, nnd = [fl1[0]], [fl2[0]]
            p1 = p2 = 1
            while p1 < len(fl1) or p2 < len(fl2):
                f1 = fl1[p1] if p1 < len(fl1) else None
                f2 = fl2[p2] if p2 < len(fl2) else None
                if f1 == f2:
                    p1 += 1; p2 += 1; continue;
                if f2 is None or (f1 is not None and f1 < f2):
                    nnn.append(f1); p1 += 1;
                else:
                    nnd.append(f2); p2 += 1;
            return self.__class__([nnn], [nnd])
        return self.__class__(numer, denom)

    __rmul__ = __mul__

    def inv(self):
        if self == 0: return 0
        return self.__class__(self.denom, self.numer)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return self * other.inv()
        return self.__class__(self.numer / other, self.denom)

    def __rtruediv__(self, other):
        return self.__class__(other * self.denom, self.numer)

    def __neg__(self):
        return self.__class__(-self.numer, self.denom)

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power, modulo=None):
        if power < 0:
            *_, last = power_supply(self, -power)
            return 1 / last
        *_, last = power_supply(self, power)
        return last

    def __str__(self):
        numer_str = f"({self.numer})" if len(self.numer) > 1 else f"{self.numer}"
        if self.denom.args == [[1]]:
            return numer_str
        denom_str = f"({self.denom})" if len(self.denom) > 1 else f"{self.denom}"
        return f"(({numer_str}) / ({denom_str}))"

    def tosympy(self):
        """ Return a sympy version of this Polynomial. """
        return self.numer.tosympy() / self.denom.tosympy()

    def __bool__(self):
        return self.numer.__bool__()
