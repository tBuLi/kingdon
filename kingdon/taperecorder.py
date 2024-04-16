from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod


@dataclass(init=False)
class TapeRecorder:
    #TODO: Create common baseclass for TapeRecorder and Multivector?
    algebra: "Algebra"
    expr: str
    _keys: tuple = field(default_factory=tuple)

    def __new__(cls, algebra, expr, keys):
        obj = object.__new__(cls)
        obj.algebra = algebra
        obj.expr = expr
        obj._keys = keys
        return obj

    def keys(self):
        return self._keys

    @cached_property
    def type_number(self) -> int:
        return int(''.join('1' if i in self.keys() else '0' for i in reversed(self.algebra.canon2bin.values())), 2)

    def __getattr__(self, basis_blade):
        bin_blade = self.algebra.canon2bin[basis_blade]
        try:
            idx = self.keys().index(bin_blade)
        except ValueError:
            return self.__class__(
                algebra=self.algebra,
                expr=f"(0,)",
                keys=(0,)
            )
        else:
            return self.__class__(
                algebra=self.algebra,
                expr=f"({self.expr}[{idx}],)",
                keys=(self.keys()[idx],)
            )

    def grade(self, *grades):
        if len(grades) == 1 and isinstance(grades[0], tuple):
            grades = grades[0]

        basis_blades = self.algebra.indices_for_grades[grades]
        indices_keys = [(idx, k) for idx, k in enumerate(self.keys()) if k in basis_blades]
        indices, keys = zip(*indices_keys) if indices_keys else (tuple(), tuple())
        expr = f"[{self.expr}[idx] for idx in {indices}]"
        return self.__class__(
            algebra=self.algebra,
            expr=expr,
            keys=keys,
        )


    def __str__(self):
        return self.expr

    def binary_operator(self, other, operator: str):
        if not isinstance(other, self.__class__):
            # Assume scalar
            keys_out, func = getattr(self.algebra, operator)[self.keys(), (0,)]
            expr = f'{func.__name__}({self.expr}, ({other},))'
        else:
            keys_out, func = getattr(self.algebra, operator)[self.keys(), other.keys()]
            expr = f'{func.__name__}({self.expr}, {other.expr})'
        return self.__class__(algebra=self.algebra, expr=expr, keys=keys_out)

    def unary_operator(self, operator: str):
        keys_out, func = getattr(self.algebra, operator)[self.keys()]
        expr = f'{func.__name__}({self.expr})'
        return self.__class__(algebra=self.algebra, expr=expr, keys=keys_out)

    # Binary operators
    gp = __mul__ = __rmul__ = partialmethod(binary_operator, operator='gp')
    sw = __rshift__ = partialmethod(binary_operator, operator='sw')
    cp = partialmethod(binary_operator, operator='cp')
    acp = partialmethod(binary_operator, operator='acp')
    ip = __or__ = partialmethod(binary_operator, operator='ip')
    sp = partialmethod(binary_operator, operator='sp')
    lc = partialmethod(binary_operator, operator='lc')
    rc = partialmethod(binary_operator, operator='rc')
    op = __xor__ = __rxor__ = partialmethod(binary_operator, operator='op')
    rp = __and__ = partialmethod(binary_operator, operator='rp')
    proj = __matmul__ = partialmethod(binary_operator, operator='proj')
    add = __add__ = __radd__ = partialmethod(binary_operator, operator='add')
    sub = __sub__ = partialmethod(binary_operator, operator='sub')
    def __rsub__(self, other): return other + (-self)
    __truediv__ = div = partialmethod(binary_operator, operator='div')

    def __pow__(self, power, modulo=None):
        if power == 0:
            return self.__class__(self.algebra, expr='(1,)', keys=(0,))

        res = self
        for i in range(1, power):
            res = res.gp(self)
        return res

    # Unary operators
    inv = partialmethod(unary_operator, operator='inv')
    neg = __neg__ = partialmethod(unary_operator, operator='neg')
    reverse = __invert__ = partialmethod(unary_operator, operator='reverse')
    involute = partialmethod(unary_operator, operator='involute')
    conjugate = partialmethod(unary_operator, operator='conjugate')
    sqrt = partialmethod(unary_operator, operator='sqrt')
    polarity = partialmethod(unary_operator, operator='polarity')
    unpolarity = partialmethod(unary_operator, operator='unpolarity')
    hodge = partialmethod(unary_operator, operator='hodge')
    unhodge = partialmethod(unary_operator, operator='unhodge')
    normsq = partialmethod(unary_operator, operator='normsq')
    outerexp = partialmethod(unary_operator, operator='outerexp')
    outersin = partialmethod(unary_operator, operator='outersin')
    outercos = partialmethod(unary_operator, operator='outercos')
    outertan = partialmethod(unary_operator, operator='outertan')

    def dual(self, kind='auto'):
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self.polarity()
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.hodge()
        elif kind == 'auto':
            raise Exception('Cannot select a suitable dual in auto mode for this algebra.')
        else:
            raise ValueError(f'No dual found for kind={kind}.')

    def undual(self, kind='auto'):
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self.unpolarity()
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.unhodge()
        elif kind == 'auto':
            raise Exception('Cannot select a suitable undual in auto mode for this algebra.')
        else:
            raise ValueError(f'No undual found for kind={kind}.')
