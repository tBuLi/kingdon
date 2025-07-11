import operator
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce, cached_property
from typing import Generator, ClassVar
from itertools import product
import re
import math
import sys

from sympy import Expr, Symbol, sympify, sinc, cos

from kingdon.codegen import _lambdify_mv
from kingdon.polynomial import RationalPolynomial

if sys.version_info >= (3, 10):
    _bit_count = int.bit_count
else:
    def _bit_count(n):
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count


@dataclass(init=False)
class MultiVector:
    algebra: "Algebra"
    _values: list = field(default_factory=list)
    _keys: tuple = field(default_factory=tuple)

    # Make MultiVector "primary" operand in operations involving ndarray.
    # (forces reflected (swapped) operands operations, like __radd__)
    __array_priority__: ClassVar[int] = 1

    def __copy__(self):
        return self.fromkeysvalues(self.algebra, self._keys, self._values)

    def __deepcopy__(self, memo):
        return self.fromkeysvalues(self.algebra, self._keys, deepcopy(self._values))

    def __new__(cls, algebra: "Algebra", values=None, keys=None, *, name=None, grades=None, symbolcls=Symbol, **items):
        """
        :param algebra: Instance of :class:`~kingdon.algebra.Algebra`.
        :param keys: Keys corresponding to the basis blades in either binary rep or as strings, e.g. :code:'"e12"'.
        :param values: Values of the multivector. If keys are provided, then keys and values should
            satisfy :code:`len(keys) == len(values)`. If no keys nor grades are provided, :code:`len(values)`
            should equal :code:`len(algebra)`, i.e. a full multivector. If grades is provided,
            then :code:`len(values)` should be identical to the number of values in a multivector
            of that grade.
        :param name: Base string to be used as the name for symbolic values.
        :param grades: Optional, :class:`tuple` of grades in this multivector.
            If present, :code:`keys` is checked against these grades.
        :param symbolcls: Optional, class to be used for symbol creation. This is a :class:`sympy.Symbol` by default,
            but could be e.g. :class:`symfit.Variable` or :class:`symfit.Parameter` when the goal is to use this
            multivector in a fitting problem.
        :param items: keyword arguments can be used to initiate multivectors as well, e.g.
            :code:`MultiVector(alg, e12=1)`. Mutually exclusive with `values` and `keys`.
        """
        if items and keys is None and values is None:
            for key in list(items.keys()):
                if key not in algebra.canon2bin:
                    target, swaps = algebra._blade2canon(key)
                    if swaps % 2:
                        items[target] = - items.pop(key)

            keys, values = zip(*((blade, items[blade]) for blade in algebra.canon2bin if blade in items))
            values = list(values)

        # Sanitize input
        if keys is not None and not all(isinstance(k, int) for k in keys):
            keys = tuple(k if k in algebra.bin2canon else algebra.canon2bin[k] for k in keys)
        if grades is None and name and keys is not None:
            grades = tuple(sorted({_bit_count(k) for k in keys}))
        values = values if values is not None else list()
        keys = keys if keys is not None else tuple()

        if grades is not None:
            if not all(0 <= grade <= algebra.d for grade in grades):
                raise ValueError(f'Each grade in `grades` needs to be a value between 0 and {algebra.d}.')
        else:
            if keys:
                grades = tuple(sorted({format(k, 'b').count('1') for k in keys}))
            else:
                grades = tuple(range(algebra.d + 1))

        if algebra.graded and keys and len(keys) != sum(math.comb(algebra.d, grade) for grade in grades):
            raise ValueError(f"In graded mode, the number of keys should be equal to "
                             f"those expected for a multivector of {grades=}.")

        # Construct a new MV on the basis of the kind of input we received.
        if isinstance(values, Mapping):
            keys, values = zip(*values.items()) if values else (tuple(), list())
            values = list(values)
        elif len(values) == sum(math.comb(algebra.d, grade) for grade in grades) and not keys:
            keys = tuple(algebra.indices_for_grades(grades))
        elif name and not values:
            # values was not given, but we do have a name. So we are in symbolic mode.
            keys = tuple(algebra.indices_for_grades(grades)) if not keys else keys
            values = list(symbolcls(f'{name}{algebra.bin2canon[k][1:]}') for k in keys)
        elif len(keys) != len(values):
            raise TypeError(f'Length of `keys` and `values` have to match.')

        if not all(isinstance(k, int) for k in keys):
            keys = tuple(key if key in algebra.bin2canon else algebra.canon2bin[key]
                         for key in keys)

        if any(isinstance(v, str) for v in values):
            values = list(val if not isinstance(val, str) else sympify(val)
                          for val in values)

        if not all(_bit_count(k) in grades for k in keys):
            raise ValueError(f"All keys should be of grades {grades}.")

        return cls.fromkeysvalues(algebra, keys, values)

    @classmethod
    def fromkeysvalues(cls, algebra, keys, values):
        """
        Initiate a multivector from a sequence of keys and a sequence of values.
        """
        obj = object.__new__(cls)
        obj.algebra = algebra
        obj._values = values
        obj._keys = keys
        return obj

    @classmethod
    def frommatrix(cls, algebra, matrix):
        """
        Initiate a multivector from a matrix. This matrix is assumed to be
        generated by :class:`~kingdon.multivector.MultiVector.asmatrix`, and
        thus we only read the first column of the input matrix.
        """
        obj = cls(algebra=algebra, values=matrix[..., 0])
        return obj

    def keys(self):
        return self._keys

    def values(self):
        return self._values

    def items(self):
        return zip(self._keys, self._values)

    def __len__(self):
        return len(self._values)

    @cached_property
    def type_number(self) -> int:
        return int(''.join('1' if i in self.keys() else '0' for i in reversed(self.algebra.canon2bin.values())), 2)


    def itermv(self, axis=None) -> Generator["MultiVector", None, None]:
        """
        Returns an iterator over the multivectors within this multivector, if it is a multidimensional multivector.
        For example, if you have a pointcloud of N points, itermv will iterate over these points one at a time.

        :param axis: Axis over which to iterate. Default is to iterate over all possible mv.
        """
        shape = self.shape[1:]
        if not shape:
            return self
        elif axis is None:
            return (
                self[indices]
                for indices in product(*(range(n) for n in shape))
            )
        else:
            raise NotImplementedError

    @property
    def shape(self):
        """ Return the shape of the .values() attribute of this multivector. """
        if hasattr(self._values, 'shape'):
            return self._values.shape
        elif hasattr(self._values[0], 'shape'):
            return len(self), *self._values[0].shape
        else:
            return len(self),

    @cached_property
    def grades(self):
        """ Tuple of the grades present in `self`. """
        return tuple(sorted({bin(ind).count('1') for ind in self.keys()}))

    def grade(self, *grades):
        """
        Returns a new  :class:`~kingdon.multivector.MultiVector` instance with
        only the selected `grades` from `self`.

        :param grades: tuple or ints, grades to select.
        """
        if len(grades) == 1 and isinstance(grades[0], tuple):
            grades = grades[0]

        items = {k: v for k, v in self.items() if _bit_count(k) in grades}
        return self.fromkeysvalues(self.algebra, tuple(items.keys()), list(items.values()))

    @cached_property
    def issymbolic(self):
        """ True if this mv contains Symbols, False otherwise. """
        symbol_classes = (Expr, RationalPolynomial)
        if self.algebra.codegen_symbolcls:
            # Allowed symbol classes. codegen_symbolcls might refer to a constructor (method): get the class instead.
            symbolcls = self.algebra.codegen_symbolcls
            symbol_classes = (*symbol_classes, symbolcls.__self__ if hasattr(symbolcls, '__self__') else symbolcls)
        return any(isinstance(v, symbol_classes) for v in self.values())

    def neg(self):
        return self.algebra.neg(self)

    __neg__ = neg

    def __invert__(self):
        """ Reversion """
        return self.algebra.reverse(self)

    def reverse(self):
        """ Reversion """
        return self.algebra.reverse(self)

    def involute(self):
        """ Main grade involution. """
        return self.algebra.involute(self)

    def conjugate(self):
        """ Clifford conjugation: involution and reversion combined. """
        return self.algebra.conjugate(self)

    def sqrt(self):
        return self.algebra.sqrt(self)

    def normsq(self):
        return self.algebra.normsq(self)

    def norm(self):
        normsq = self.normsq()
        return normsq.sqrt()

    def normalized(self):
        """ Normalized version of this multivector. """
        return self / self.norm()

    def inv(self):
        """ Inverse of this multivector. """
        return self.algebra.inv(self)

    def add(self, other):
        return self.algebra.add(self, other)

    __radd__ = __add__ = add

    def sub(self, other):
        return self.algebra.sub(self, other)

    __sub__ = sub

    def __rsub__(self, other):
        return self.algebra.sub(other, self)

    def div(self, other):
        return self.algebra.div(self, other)

    __truediv__ = div

    def __rtruediv__(self, other):
        return self.algebra.div(other, self)

    def __str__(self):
        if not len(self.values()):
            return '0'

        def print_value(val):
            s = str(val)
            if isinstance(val, Expr):
                if val.is_Symbol:
                    return s
                return f"({s})"
            if isinstance(val, float):
                return f'{val:.3}'
            if isinstance(val, int):
                return s
            if bool(re.search(r'[\(\[\{].*[\)\]\}]$', s)):
                # If the expression already has brackets, like numpy arrays
                return s
            return f'({s})'

        def print_key(blade):
            if blade == 'e':
                return '1'
            return self.algebra.pretty_blade + ''.join(self.algebra.pretty_digits[num] for num in blade[1:])

        canon_sorted_vals = {print_key(self.algebra.bin2canon[key]): val
                             for key, val in self.items()}
        str_repr = ' + '.join(
            [f'{print_value(val)} {blade}' if blade != '1' else f'{print_value(val)}'
             for blade, val in canon_sorted_vals.items() if (val.any() if hasattr(val, 'any') else val)]
        )
        return str_repr

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            p.text(str(self))

    def __format__(self, format_spec):
        if format_spec == 'keys_binary':
            return bin(self.type_number)[2:].zfill(len(self.algebra))
        return str(self)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)

        values = self.values()
        if isinstance(values, (tuple, list)):
            return_values = values.__class__(value[item] for value in values)
        else:
            return_values = values[(slice(None), *item)]
        return self.__class__.fromkeysvalues(self.algebra, keys=self.keys(), values=return_values)

    def __setitem__(self, indices, values):
        if isinstance(values, MultiVector):
            if self.keys() != values.keys():
                raise ValueError('setitem with a multivector is only possible for equivalent MVs.')
            values = values.values()

        if not isinstance(indices, tuple):
            indices = (indices,)

        if isinstance(self.values(), (tuple, list)):
            for self_values, other_value in zip(self.values(), values):
                self_values[indices] = other_value
        else:
            self.values()[(slice(None), *indices)] = values

    def __getattr__(self, basis_blade):
        # TODO: if this first check is not true, raise hell instead?
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            raise AttributeError(f'{self.__class__.__name__} object has no attribute or basis blade {basis_blade}')
        basis_blade, swaps = self.algebra._blade2canon(basis_blade)
        if basis_blade not in self.algebra.canon2bin:
            return 0
        try:
            idx = self.keys().index(self.algebra.canon2bin[basis_blade])
        except ValueError:
            return 0
        return self._values[idx] if swaps % 2 == 0 else - self._values[idx]

    def __setattr__(self, basis_blade, value):
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            return super().__setattr__(basis_blade, value)
        if (key := self.algebra.canon2bin[basis_blade]) in self.keys():
            self._values[key] = value
        else:
            raise TypeError("The keys of a MultiVector are immutable, please create a new MultiVector.")

    def __delattr__(self, basis_blade, value):
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            return super().__setattr__(basis_blade, value)
        raise TypeError("The keys of a MultiVector are immutable, please create a new MultiVector.")

    def __contains__(self, item):
        item = item if isinstance(item, int) else self.algebra.canon2bin[item]
        return item in self._keys

    def __bool__(self):
        return bool(self.values())

    @cached_property
    def free_symbols(self) -> set:
        return reduce(operator.or_, (v.free_symbols for v in self.values() if hasattr(v, "free_symbols")), set())

    def map(self, func) -> "MultiVector":
        """
        Returns a new multivector where `func` has been applied to all the values.
        If `func` has one argument, it is called on each entry of self.values().
        If `func` has two arguments, the function is called with the key, value pairs as per
        self.items() instead.
        """
        if hasattr(func, '__code__') and func.__code__.co_argcount == 2:
            vals = [func(k, v) for k, v in self.items()]
        else:
            vals = [func(v) for v in self.values()]
        return self.fromkeysvalues(self.algebra, keys=self.keys(), values=vals)

    def filter(self, func=None) -> "MultiVector":
        """
        Returns a new multivector containing only those elements for which `func` was true-ish.
        If no function was provided, use the simp_func of the Algebra.
        If `func` has one argument, it is called on each entry of self.values().
        If `func` has two arguments, the function is called with the key, value pairs as per
        self.items() instead.
        """
        if func is None:
            func = self.algebra.simp_func
        if hasattr(func, '__code__') and func.__code__.co_argcount == 2:
            keysvalues = tuple((k, v) for k, v in self.items() if func(k, v))
        else:
            keysvalues = tuple((k, v) for k, v in self.items() if func(v))
        if not keysvalues:
            return self.fromkeysvalues(self.algebra, keys=tuple(), values=list())
        keys, values = zip(*keysvalues)
        return self.fromkeysvalues(self.algebra, keys=keys, values=list(values))

    @cached_property
    def _callable(self):
        """ Return the callable function for this MV. """
        return _lambdify_mv(self)

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise Exception('Please provide all input either as positional arguments or as keywords arguments, not both.')

        if not self.free_symbols:
            return self
        keys_out, func = self._callable
        if kwargs:
            args = [v for k, v in sorted(kwargs.items(), key=lambda x: x[0])]
        values = func(args)
        return self.fromkeysvalues(self.algebra, keys_out, values)

    def asmatrix(self):
        """ Returns a matrix representation of this multivector. """
        bin2index = {k: i for i, k in enumerate(self.algebra.canon2bin.values())}
        return sum(v * self.algebra.matrix_basis[bin2index[k]] for k, v in self.items())

    def asfullmv(self, canonical=True):
        """
        Returns a full version of the same multivector.

        :param canonical: If True (default) the values are in canonical order,
          even if the mutivector was already dense.
        """
        if canonical:
            keys = self.algebra.indices_for_grades(tuple(range(self.algebra.d + 1)))
        else:
            keys = tuple(range(len(self.algebra)))
        values = [getattr(self, self.algebra.bin2canon[k]) for k in keys]
        return self.fromkeysvalues(self.algebra, keys=keys, values=values)

    def gp(self, other):
        return self.algebra.gp(self, other)

    __mul__ = gp

    def __rmul__(self, other):
        return self.algebra.gp(other, self)

    def sw(self, other):
        """
        Apply :code:`x := self` to :code:`y := other` under conjugation:
        :code:`x.sw(y) = x*y*~x`.
        """
        return self.algebra.sw(self, other)

    __rshift__ = sw

    def __rrshift__(self, other):
        return self.algebra.sw(other, self)

    def proj(self, other):
        """
        Project :code:`x := self` onto :code:`y := other`: :code:`x @ y = (x | y) * ~y`.
        For correct behavior, :code:`x` and :code:`y` should be normalized (k-reflections).
        """
        return self.algebra.proj(self, other)

    __matmul__ = proj

    def __rmatmul__(self, other):
        return self.algebra.proj(other, self)

    def cp(self, other):
        """
        Calculate the commutator product of :code:`x := self` and :code:`y := other`:
        :code:`x.cp(y) = 0.5*(x*y-y*x)`.
        """
        return self.algebra.cp(self, other)

    def acp(self, other):
        """
        Calculate the anti-commutator product of :code:`x := self` and :code:`y := other`:
        :code:`x.cp(y) = 0.5*(x*y+y*x)`.
        """
        return self.algebra.acp(self, other)

    def ip(self, other):
        return self.algebra.ip(self, other)

    __or__ = ip

    def __ror__(self, other):
        return self.algebra.ip(other, self)

    def op(self, other):
        return self.algebra.op(self, other)

    __xor__ = __rxor__ = op

    def lc(self, other):
        return self.algebra.lc(self, other)

    def rc(self, other):
        return self.algebra.rc(self, other)

    def sp(self, other):
        r""" Scalar product: :math:`\langle x \cdot y \rangle`. """
        return self.algebra.sp(self, other)

    def rp(self, other):
        return self.algebra.rp(self, other)

    __and__ = rp

    def __rand__(self, other):
        return self.algebra.rp(other, self)

    def __pow__(self, power, modulo=None):
        # TODO: this should also be taken care of via codegen, but for now this workaround is ok.
        if power == 0:
            return self.algebra.scalar((1,))
        elif power < 0:
            res = x = self.inv()
            power *= -1
        else:
            res = x = self

        if power == 0.5:
            return res.sqrt()

        for i in range(1, power):
            res = res.gp(x)
        return res

    def outerexp(self):
        return self.algebra.outerexp(self)

    def outersin(self):
        return self.algebra.outersin(self)

    def outercos(self):
        return self.algebra.outercos(self)

    def outertan(self):
        return self.algebra.outertan(self)

    def exp(self, cosh=None, sinhc=None, sqrt=None):
        r"""
        Calculate the exponential of simple elements, meaning an element that squares to a scalar.
        Works for python float, int and complex dtypes, and for symbolic expressions using sympy.
        For more control, it is possible to explicitly provide a `cosh`, `sinhc`, and `sqrt` function.
        If you provide one, you must provide all.

        The argument to `sqrt` is the scalar :math:`s = \langle x^2 \rangle_0`, while the input to the
        `cosh` and `sinhc` functions is the output of the sqrt function applied to :math:`s`.

        For example, for a simple rotation `kingdon`'s implementation is equivalent to

        .. code-block ::

            alg = Algebra(2)
            x = alg.bivector(e12=1)
            x.exp(
                cosh=np.cos,
                sinhc=np.sinc,
                sqrt=lambda s: (-s)**0.5,
            )
        """
        ll = (self * self).filter()
        if ll.grades and ll.grades != (0,):
            raise NotImplementedError(
                'Currently only elements that square to a scalar (i.e. are simple) can be exponentiated.'
            )

        ll = ll.e
        if sqrt is None and cosh is None and sinhc is None:
            if isinstance(ll, Expr):
                sqrt = lambda x: (-x) ** 0.5
                cosh = cos
                sinhc = sinc
            elif isinstance(ll, (float, int)) and ll > 0:
                sqrt = lambda x: x ** 0.5
                import numpy as np
                cosh = np.cosh
                sinhc = lambda x: np.sinh(x) / x
            elif isinstance(ll, (float, int)) and ll == 0:
                sqrt = lambda x: x ** 0.5
                import numpy as np
                cosh = sinhc = lambda x: 1
            else:
                # Assume numpy
                sqrt = lambda x: (-x) ** 0.5
                import numpy as np
                cosh = np.cos
                sinhc = lambda x: np.sinc(x / np.pi)

        l = sqrt(ll)
        return self * sinhc(l) + cosh(l)

    def polarity(self):
        return self.algebra.polarity(self)

    def unpolarity(self):
        return self.algebra.unpolarity(self)

    def hodge(self):
        return self.algebra.hodge(self)

    def unhodge(self):
        return self.algebra.unhodge(self)

    def dual(self, kind='auto'):
        """
        Compute the dual of `self`. There are three different kinds of duality in common usage.
        The first is polarity, which is simply multiplying by the inverse PSS from the right. This is the only game in
        town for non-degenerate metrics (Algebra.r = 0). However, for degenerate spaces this no longer works, and we
        have two popular options: Poincaré and Hodge duality.

        By default, :code:`kingdon` will use polarity in non-degenerate spaces, and Hodge duality for spaces with
        `Algebra.r = 1`. For spaces with `r > 2`, little to no literature exists, and you are on your own.

        :param kind: if 'auto' (default), :code:`kingdon` will try to determine the best dual on the
            basis of the signature of the space. See explenation above.
            To ensure polarity, use :code:`kind='polarity'`, and to ensure Hodge duality,
            use :code:`kind='hodge'`.
        """
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self.polarity()
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.hodge()
        elif kind == 'auto':
            raise Exception('Cannot select a suitable dual in auto mode for this algebra.')
        else:
            raise ValueError(f'No dual found for kind={kind}.')

    def undual(self, kind='auto'):
        """
        Compute the undual of `self`. See :class:`~kingdon.multivector.MultiVector.dual` for more information.
        """
        if kind == 'polarity' or kind == 'auto' and self.algebra.r == 0:
            return self.unpolarity()
        elif kind == 'hodge' or kind == 'auto' and self.algebra.r == 1:
            return self.unhodge()
        elif kind == 'auto':
            raise Exception('Cannot select a suitable undual in auto mode for this algebra.')
        else:
            raise ValueError(f'No undual found for kind={kind}.')
