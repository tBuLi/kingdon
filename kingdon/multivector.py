import operator
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce, cached_property, wraps
from typing import ClassVar
from types import EllipsisType
from itertools import product
import re
import math


from sympy import Expr, Symbol, sympify, sinc, cos
from sympy.utilities.iterables import iterable

import kingdon.operators as ops
from kingdon.polynomial import RationalPolynomial


class MultiVectorType(type):
    """
    MultiVector type allows typehinting for MultiVectors of a given shape.
    For example, :code:`MultiVector[3]` is interpreted as a MultiVectors of shape (N, 3) by :code:`Algebra.compile`,
    where N is the number of blades in the multivector.
    """
    def __getitem__(cls, item): return cls, item


@dataclass(init=False)
class MultiVector(metaclass=MultiVectorType):
    algebra: "Algebra"
    _values: list = field(default_factory=list)
    _keys: tuple[int] = field(default_factory=tuple)

    # Make MultiVector "primary" operand in operations involving ndarray.
    # (forces reflected (swapped) operands operations, like __radd__)
    __array_priority__: ClassVar[int] = 1

    def __copy__(self):
        return self.fromkeysvalues(self.algebra, self._keys, self._values, raw=True)

    def __deepcopy__(self, memo):
        return self.fromkeysvalues(self.algebra, self._keys, deepcopy(self._values), raw=True)

    def __new__(cls, algebra: "Algebra", values=None, keys=None, *, name=None, grades=None, symbolcls=None, **items):
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
        if name is not None:
            if values is not None or items:
                raise ValueError('Cannot provide both name and values.')
            return cls.fromname(algebra, name, keys=keys, grades=grades, symbolcls=symbolcls)

        if isinstance(values, Mapping):
            items = values; values = None
        if items:
            if keys is not None or values is not None:
                raise ValueError('Cannot provide both items and keys or values.')
            for key in list(items.keys()):
                if isinstance(key, str):
                    if not re.match(r'^e[0-9a-fA-Z]*$', key):
                        raise KeyError(f'The key {key} does not refer to a valid basis blade.')
                    target, swaps = algebra._blade2canon(key)
                    items[target] = - items.pop(key) if swaps % 2 else items.pop(key); key = target
                    items[algebra.canon2bin[key]] = items.pop(key)  # Switch to binary key

            keys, values = zip(*((key, items[key]) for key in algebra.canon2bin.values() if key in items))
            values = list(values)

        keys = keys if keys is not None else tuple() if values is None else None
        values = values if values is not None else list()
        if isinstance(values, tuple):  # Values are always a list, e.g. so they can be updated inplace.
            values = list(values)
        if any(isinstance(v, str) for v in values):
            converter = symbolcls or sympify
            values = list(val if not isinstance(val, str) else converter(val)
                          for val in values)
        if grades and not all(0 <= grade <= algebra.d for grade in grades):
            raise ValueError(f'Each grade in `grades` needs to be a value between 0 and {algebra.d}.')
        keys = cls.sanitize_keys_grades(algebra, keys, grades)
        if len(keys) != len(values):
            raise TypeError(f'Length of `keys` and `values` have to match.')
        inst = cls.fromkeysvalues(algebra, keys, values, raw=cls._issymbolic(algebra, values))
        return inst

    @classmethod
    def fromkeysvalues(cls, algebra: "Algebra", keys: tuple, values: Sequence, values_asarray=None, raw=False):
        """
        Initiate a multivector from a sequence of keys and a sequence of values.
        All array construction ultimately funnels through this function.

        :param algebra: :class:`~kingdon.algebra.Algebra`
        :param keys: Keys corresponding to the basis blades in binary rep.
        :param values: Values of the multivector.
        :param values_asarray: asarray function to be applied to values. E.g. numpy.asarray or torch.asarray. Defaults to :code:`Algebra.values_asarray`.
        :param raw: values_asarray application is skipped.
        """
        if not raw and isinstance(values, (list, tuple)):
            values_asarray = values_asarray or algebra.values_asarray
            if type(values) is not values_asarray:
                values = values_asarray(values)
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

    @classmethod
    def sanitize_keys_grades(cls, algebra, keys=None, grades=None) -> tuple[int, tuple[int]]:
        """
        Ensure that keys and grades are in agreement with the layout for ``cls``.
        If no keys or grades are provided, they are created.
        """
        layout = algebra._type_layouts.get(cls, {})
        if keys is None:
            # Generate keys from layout. Since they are generated from layout, we don't need to validate them against layout.
            if layout:
                if grades is None:
                    keys = tuple(k for k, v in layout.items() if v == ...)
                else:
                    keys = tuple(k for k, v in layout.items() if v == ... and k.bit_count() in grades)
                return keys

            if grades is None:
                grades = tuple(range(algebra.d + 1))
            keys = tuple(algebra.indices_for_grades(grades))
        else:
            if not all(isinstance(k, int) for k in keys):  # Not done in one loop because then we would always create a new keys tuple even if it is already all ints.
                keys = tuple(key if isinstance(key, int) else algebra.canon2bin[key] for key in keys)

        # Validate keys against layout if one is provided.
        if layout:
            if not all(layout.get(k) == ... for k in keys):
                raise TypeError(f'The provided keys {keys} are not free variables for {cls.__name__} with layout {layout}.')
            if grades is None:
                grades = tuple(sorted({k.bit_count() for k in keys + tuple(k for k, v in layout.items() if v != ...)}))

        if algebra.full_layout and algebra._type_layouts:  # The second condition is false before layouts have been bound.
            if layout and len(keys) != len([v for v in layout.values() if v == ...]):
                raise ValueError(f"In full_layout mode, the number of keys should be equal to "
                                 f"those expected for a {cls} with layout {layout=}.")
            elif not layout and len(keys) != len(algebra):
                raise ValueError(f"In full_layout mode, the number of keys for {cls} should be equal to {len(algebra)}.")

        return keys

    @classmethod
    def fromname(cls, algebra, name: str, keys=None, grades=None, symbolcls=None):
        """
        Initiate a symbolic multivector.
        """
        if symbolcls is None:
            symbolcls = algebra.symbolcls or Symbol
        keys = cls.sanitize_keys_grades(algebra, keys, grades)
        values = list(symbolcls(f'{name}{algebra.bin2canon[k][1:]}') for k in keys)
        instance = cls.fromkeysvalues(algebra, keys, values, raw=True)
        return instance

    def keys(self) -> tuple:
        return self._keys

    def values(self) -> list:
        return self._values

    def items(self):
        return zip(self._keys, self._values)

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized 0-dimensional multivector')
        return self.shape[0]

    def __iter__(self):
        """
        Iterate over the multivector along the first axis of :attr:`shape`. E.g. iterating
        over a pointcloud with shape :code:`(N,)` yields those N points one at a time.
        """
        if not self.shape:
            raise TypeError(f'0-dimensional {self.__class__.__name__} is not iterable')
        return (self[i] for i in range(self.shape[0]))

    @property
    def ndim(self):
        """ Number of array dimensions of this multivector. """
        return len(self.shape)

    @cached_property
    def type_number(self) -> int:
        return int(''.join('1' if i in self.keys() else '0' for i in reversed(self.algebra.canon2bin.values())), 2)

    @cached_property
    def shape(self) -> tuple:
        """ Return the shape of the .values() attribute of this multivector. """
        def _list_shape(v):
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
                inner = _list_shape(v[0])
                if all(isinstance(w, (list, tuple)) and len(w) == len(v[0]) for w in v[1:]):
                    return (len(v), *inner)
            return (len(v),)

        if hasattr(self._values, 'shape'):
            return self._values.shape[1:]
        if self._values:
            first = self._values[0]
            if hasattr(first, 'shape') and all(getattr(v, 'shape', None) == first.shape for v in self._values[1:]):
                return first.shape
            if isinstance(first, (list, tuple)) and all(isinstance(v, (list, tuple)) and len(v) == len(first) for v in self._values[1:]):
                return _list_shape(first)
        return ()

    @property
    def type_layout(self) -> dict[int, float | EllipsisType]:
        r"""
        Layout of :code:`type(self)`: a mapping from blade binary key (int) to either
        :code:`...` for a free component or to the numerical value of a structural constant.
        For example, a point in :code:`Algebra.fromname("2DPGA")` has layout
        :code:`{'e20': ..., 'e01': ..., 'e21': 1.0}`.
        E.g. the :code:`1.0` a normalized point has on :math:`\mathbf{e}_0^*`.
        Types without a layout return an empty dict.
        """
        return self.algebra._type_layouts.get(type(self), {})

    @cached_property
    def grades(self):
        """ Tuple of the grades present in `self`. """
        grades_in_keys = {k.bit_count() for k in self.keys()}
        grades_in_fixed_layout = {k.bit_count() for k, v in self.type_layout.items() if v != ...}
        return tuple(sorted(grades_in_keys | grades_in_fixed_layout))

    def grade(self, *grades):
        """
        Returns a new  :class:`~kingdon.multivector.MultiVector` instance with
        only the selected `grades` from `self`.

        :param grades: tuple or ints, grades to select.
        """
        if len(grades) == 1 and isinstance(grades[0], tuple):
            grades = grades[0]

        items = {k: v for k, v in self.items() if k.bit_count() in grades}
        res_layout = {k: v for k, v in self.type_layout.items() if k.bit_count() in grades}
        res_layout.update({k: ... for k in items})
        if res_layout:
            from .codegen import resolve_layout
            MVType, _ = resolve_layout(self.algebra._type_layouts, res_layout, default=self.algebra.mvtype)
        else:
            MVType = self.algebra.mvtype
        return MVType.fromkeysvalues(self.algebra, tuple(items.keys()), list(items.values()), raw=self.issymbolic)

    @staticmethod
    def _issymbolic(algebra, values) -> bool:
        """ True if any of the `values` is a Symbol, False otherwise. """
        symbol_classes = (Expr, RationalPolynomial)
        if algebra.codegen_symbolcls:
            # Allowed symbol classes. codegen_symbolcls might refer to a constructor (method): get the class instead.
            symbolcls = algebra.codegen_symbolcls
            symbol_classes = (*symbol_classes, symbolcls.__self__ if hasattr(symbolcls, '__self__') else symbolcls)
        return any(isinstance(v, symbol_classes) for v in values)

    @cached_property
    def issymbolic(self):
        """ True if this mv contains Symbols, False otherwise. """
        return self._issymbolic(self.algebra, self.values())

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
        layout = self.type_layout

        if not len(self.values()) and not layout:
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

        if layout:
            self_dict = dict(self.items())
            keysvals = [(k, self_dict[k] if v is ... else v)
                        for k, v in layout.items()
                        if v is not ... or k in self_dict]
        else:
            keysvals = list(self.items())

        canon_sorted_vals = {print_key(self.algebra.bin2canon[key]): val
                             for key, val in keysvals}
        str_repr = ' + '.join(
            [f'{print_value(val)} {blade}' if blade != '1' else f'{print_value(val)}'
             for blade, val in canon_sorted_vals.items() if (val.any() if hasattr(val, 'any') else val)]
        )
        return str_repr or '0'

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
        values = self.values()
        if not isinstance(values, (tuple, list)):  # Assume it obeys the python array API
            if not isinstance(item, tuple):
                item = (item,)
            return_values = values[(slice(None), *item)]
        elif values and all(iterable(value) for value in values):
            if isinstance(values[0], (list, tuple)):  # These can only be sliced with integers.
                if isinstance(item, tuple) and len(item) == 1: item = item[0]
            return_values = values.__class__(value[item] for value in values)
        else:
            raise IndexError("Cannot index a multivector with a non-iterable value.")
        return self.__class__.fromkeysvalues(self.algebra, keys=self.keys(), values=return_values, raw=self.issymbolic)

    def __setitem__(self, indices, values: 'MultiVector'):
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

    def set(self, other: 'MultiVector') -> "Self":
        """Overwrite the values of this MV with the values of another MV."""
        if self.keys() != other.keys():
            raise ValueError('set is only possible for MVs with the same keys.')
        self._values[:] = other._values[:]
        return self

    def __getattr__(self, basis_blade):
        # TODO: if this first check is not true, raise hell instead?
        if not re.match(r'^e[0-9a-fA-Z]*$', basis_blade):
            raise AttributeError(f'{self.__class__.__name__} object has no attribute or basis blade {basis_blade}')
        basis_blade, swaps = self.algebra._blade2canon(basis_blade)
        if basis_blade not in self.algebra.canon2bin:
            return 0
        k, val = self.algebra.canon2bin[basis_blade], 0
        try:
            idx = self.keys().index(k)
            val = self._values[idx]
        except ValueError:
            if layout := self.type_layout:
                val = layout.get(k, 0)
                val = 0 if val == ... else val
        return val if swaps % 2 == 0 else - val

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
        return bool(self._keys) or any(v != ... and v for v in self.type_layout.values())

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
        return self.fromkeysvalues(self.algebra, keys=self.keys(), values=vals, raw=self.issymbolic)

    def filter(self, func=None, map=False) -> "MultiVector":
        """
        Returns a new multivector containing only those elements for which `func` was true-ish.
        If no function was provided, use the simp_func of the Algebra.
        If `func` has one argument, it is called on each entry of self.values().
        If `func` has two arguments, the function is called with the key, value pairs as per
        self.items() instead.
        If :code:`map` is true, the func is also applied as a map function at the same time.
        """
        if func is None:
            func = self.algebra.simp_func
        if hasattr(func, '__code__') and func.__code__.co_argcount == 2:
            if map: keysvalues = tuple((k, fv) for k, v in self.items() if (fv := func(k, v)))
            else:   keysvalues = tuple((k, v) for k, v in self.items() if func(k, v))
        else:
            if map: keysvalues = tuple((k, fv) for k, v in self.items() if (fv := func(v)))
            else:   keysvalues = tuple((k, v) for k, v in self.items() if func(v))
        if not keysvalues:
            return self.fromkeysvalues(self.algebra, keys=tuple(), values=list(), raw=self.issymbolic)
        keys, values = zip(*keysvalues)
        return self.fromkeysvalues(self.algebra, keys=keys, values=list(values), raw=self.issymbolic)

    def asmatrix(self):
        """ Returns a matrix representation of this multivector. """
        bin2index = {k: i for i, k in enumerate(self.algebra.canon2bin.values())}
        return sum(v * self.algebra.matrix_basis[bin2index[k]] for k, v in self.items())

    def asfullmv(self, canonical=True):
        """
        Returns a full version of the same multivector.
        Preserves the type of the multivector.

        :param canonical: If True (default) the values are in canonical order,
          even if the mutivector was already dense.
        """
        if canonical:
            keys = tuple(self.algebra.indices_for_grades(tuple(range(self.algebra.d + 1))))
        else:
            keys = tuple(range(len(self.algebra)))
        values = [getattr(self, self.algebra.bin2canon[k]) for k in keys]
        return self.fromkeysvalues(self.algebra, keys=keys, values=values)

    def asmvtype(self, MVType=None):
        """ Cast to a specific multivector type. If no type is provided, use the algebra's own. """
        MVType = MVType or self.algebra.mvtype
        if type(self) == MVType:
            return self
        if layout := self.type_layout:
            # Sort the layout to canonical order, since a layout may be in whatever order its type likes.
            layout = {k: layout[k] for k in self.algebra.canon2bin.values() if k in layout}
            keysvalues = tuple((k, v if v != ... else getattr(self, self.algebra.bin2canon[k]))
                               for k, v in layout.items() if k in self.keys() or v != ...)
            keys, values = zip(*keysvalues) if keysvalues else (tuple(), list())
            values = list(values)  # Values are always a list, e.g. so they can be updated inplace.
        else:
            keys, values = self.keys(), self.values()
        if MVType == self.algebra.mvtype:
            return MVType.fromkeysvalues(self.algebra, keys, values, raw=self.issymbolic)  # Faster than the generic constructor because it doesn't validate the input.
        return MVType(self.algebra, keys=keys, values=values)

    def gp(self, other):
        return self.algebra.gp(self, other)

    __mul__ = gp

    def __rmul__(self, other):
        return self.algebra.gp(other, self)

    def sw(self, other):
        r"""
        Apply the normalized versor (k-reflection) :code:`x := self` to the :math:`\ell`-blade:code:`y := other` under conjugation:
        :math:`x[y] = (-1)^{k \ell} x y x^{-1}`.
        If :code:`y` is a multivector instead of a blade, the formula is applied to each pure
        grade component of :code:`y` separately to ensure a consistent result.
        **Important**: note that :code:`x` is assumed to be normalized such that :math:`x \widetilde{x} = 1`
        (i.e. :code:`x.normsq() == 1`). Moreover, grade preservation is enforced by the code.
        Expect unexpected results if this operator is used with non-versors.
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

    # TODO: perhaps exp should a function not a method?
    def exp(self: "Bivector", cosh=None, sinhc=None, sqrt=None) -> "Bireflection":
        r"""
        Calculate the exponential of simple bivectors, meaning a bivector that squares to a scalar.
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
        ll = (-self.normsq()).filter()
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
                cosh = sinhc = lambda x: self.algebra.blades.e
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

    @classmethod
    def layout(cls, algebra, name):
        return algebra.mvtype.fromname(algebra, name, symbolcls=algebra.codegen_symbolcls or RationalPolynomial.fromname)


### Below are common multivector types.
class KVector(MultiVector):
    """ Baseclass for k-vectors. """
    layout_grades: ClassVar[tuple[int] | None] = None

    @classmethod
    def layout(cls, algebra, name):
        return algebra.mvtype.fromname(algebra, name, grades=cls.layout_grades,
                                       symbolcls=algebra.codegen_symbolcls or RationalPolynomial.fromname)
class Scalar(KVector): layout_grades = (0,)
class Vector(KVector): layout_grades = (1,)
class Bivector(KVector): layout_grades = (2,)
class Trivector(KVector): layout_grades = (3,)
class Quadvector(KVector): layout_grades = (4,)
class Pentavector(KVector): layout_grades = (5,)
class Hexavector(KVector): layout_grades = (6,)
class Heptavector(KVector): layout_grades = (7,)
class Octovector(KVector): layout_grades = (8,)

# k-reflections
class Bireflection(MultiVector):
    r"""
    A bireflection :math:`R` is assumed to be identical to :math:`\mathbf{R} = p \widetilde{q}`
    with :math:`p` and :math:`q` normalized vectors, such that :math:`R \widetilde{R} = 1`.
    """
    @classmethod
    def layout(cls, algebra, name):
        p = Vector.layout(algebra, f'{name}_1')
        q = Vector.layout(algebra, f'{name}_2')
        qr = ops.reverse(q)
        return ops.gp(p, qr)


### PGA types
class Direction(MultiVector):
    r"""
    PGA type. A direction is the dual of a Euclidean vector, i.e. :math:`d = d^i \mathbf{e}_i^*`.
    As such it is an ideal point and a pseudovector.
    """
    @classmethod
    def layout(cls, algebra, name):
        return ops.polarity(Vector.layout(algebra, name))


class EVector(Vector):
    r"""
    PGA type. A Euclidean vector, i.e. :math:`v = v^i \mathbf{e}_i`.
    Its dual is a direction.
    """
    @classmethod
    def layout(cls, algebra, name):
        return ops.unhodge(Direction.layout(algebra, name))


class UPoint(Vector):
    r"""
    PGA type. Undual of a point, i.e. :math:`x = \mathbf{e}_0 + x^i \mathbf{e}_i`.
    Defined such that the hodge dual of this type is of type :class:`Point`.
    """
    @classmethod
    def layout(cls, algebra, name):
        ev = EVector.layout(algebra, name)
        idx = algebra.signature.index(0) + algebra.start_index  # Find the correct index of 'e0'
        key = algebra.canon2bin[f'e{idx}']
        origin = algebra.mvtype.fromkeysvalues(algebra, (key,), [algebra.codegen_symbolcls('x') * 0 + 1], raw=True)
        return ops.add(ev, origin)


class Point(MultiVector):
    r"""
    PGA type. A point, i.e. :math:`p = \mathbf{e}_0^* + x^i \mathbf{e}_i^*`.
    Defined as the hodge dual of :class:`UPoint`, such that points can be created in a dimension agnostic way as

    .. code-block:: python

        p = alg.upoint(e1='x').dual()
    """
    @classmethod
    def layout(cls, algebra, name):
        return ops.hodge(UPoint.layout(algebra, name))


class Translation(Bireflection):
    r"""
    PGA type. A translation :math:`R` is assumed to be identical to :math:`\mathbf{R} = p \widetilde{q}`
    with :math:`p` and :math:`q` points, such that :math:`\langle R \widetilde{R} \rangle_0 = 1`.
    """
    @classmethod
    def layout(cls, algebra, name):
        p = Point.layout(algebra, f'{name}_1')
        q = Point.layout(algebra, f'{name}_2')
        qr = ops.reverse(q)
        return ops.gp(p, qr)


def _zeros_like(x):
    """
    Zeros with the same shape, dtype and device as the coefficient `x`.

    Array types offer no common constructor, so we make one out of `x` itself: give it an
    axis of length zero and then sum over that axis. Summing nothing is exactly zero, and
    since the values of `x` are never read this holds even if they are :code:`NaN`.
    :class:`~kingdon.einops_backend.KingdonBackend` does the same through the einops
    primitives, which reach the array types that do not expose the python array API.
    """
    if isinstance(x, (list, tuple)):
        return type(x)(_zeros_like(v) for v in x)  # Coefficients kept as nested python sequences.
    if not hasattr(x, 'shape'):
        return 0  # A python number or a symbol has no shape to match.
    return x[None][:0].sum(axis=0)


def _union_keys(mvs: list[MultiVector]) -> tuple[int, ...]:
    """
    The keys of a multivector able to hold all of `mvs`: the union of their keys, ordered
    like the layout of their common type, or canonically for types without a layout.

    :param mvs: multivectors, all of the same type.
    """
    keys = mvs[0].keys()
    if all(mv.keys() == keys for mv in mvs[1:]):
        return keys
    union = set().union(*(mv.keys() for mv in mvs))
    # A layout is already in the order that its type uses.
    order = list(mvs[0].type_layout) or mvs[0].algebra.canon2bin.values()
    return tuple(k for k in order if k in union)


def _coefficients(mv: MultiVector, keys: tuple[int, ...], zeros_like=_zeros_like) -> list:
    """
    The coefficients of `mv` for each of `keys`, one per key. The blades that `mv` lacks
    contribute zeros, shaped like the coefficients that it does have.

    :param zeros_like: how to make those zeros from a coefficient of `mv`.
    """
    if mv.keys() == keys:
        return list(mv.values())
    coefficients = dict(mv.items())
    if not coefficients:
        raise TypeError(f'Cannot give a {type(mv).__name__} without coefficients the keys {keys}, '
                        'since there is nothing to infer the shape of its zeros from.')
    template = next(iter(coefficients.values()))
    return [coefficients[k] if k in coefficients else zeros_like(template) for k in keys]


def stack(mvs: list[MultiVector], stack_func=None) -> MultiVector[None]:
    """
    Stack a list of multivectors along a new "first" dimension.
    All multivectors must have the same type and shape. Their keys may differ: the result
    gets the union of the keys of `mvs`, and a blade an input does not have contributes zeros.
    Remember that the first dimension of a multivector is always reserved for kingdon's multivector coefficients, so the new dimension will be the one after that.
    As a result, this function returns a multivector with shape :code:`(mvs[0].shape[0], len(mvs), *mvs[0].shape[1:])`.
    To be compatible with :code:`numpy` or :code:`torch` you can provide a custom `stack_func` that will be used to
    stack the values of the multivectors. By default this is :code:`values_asarray` of the algebra.

    For example, to stack using torch you can use::

        >>> import torch
        >>> alg = Algebra(2)
        >>> mvs = [alg.vector(torch.randn(2)) for _ in range(3)]
        >>> x = stack(mvs, stack_func=torch.stack)
        >>> x.shape
        (2, 3)

    In order to have more control over the stacking dimensions, use :code:`einops.pack` instead, like::

        >>> import einops
        >>> import kingdon.einops_backend
        >>> alg = Algebra(2)
        >>> mvs = [alg.vector(torch.randn(2, 4)) for _ in range(3)]
        >>> x, _ = einops.pack(mvs, '* n')  # n matches 4, insert new dimension to the left
        >>> x.shape
        (2, 3, 4)
        >>> y, _ = einops.pack(mvs, 'n *')  # n matches 4, insert new dimension to the right
        >>> y.shape
        (2, 4, 3)

    :param mvs: List of multivectors to stack.
    :param stack_func: Function to stack the values of the multivectors, like :code:`numpy.stack` or :code:`torch.stack`. Defaults to :code:`list`.
    :return: A new multivector with shape :code:`(len(mvs), *mvs[0].shape)`.
    """
    if not all(mv.shape == mvs[0].shape for mv in mvs[1:]):
        raise TypeError('All multivectors must have the same shape.')
    if not len(set(type(mv) for mv in mvs)) == 1:
        raise TypeError('All multivectors must have the same type.')
    if stack_func is None:
        stack_func = mvs[0].algebra.values_asarray
    keys = _union_keys(mvs)
    per_key = zip(*(_coefficients(mv, keys) for mv in mvs))
    return type(mvs[0]).fromkeysvalues(mvs[0].algebra, keys, stack_func([stack_func(vals) for vals in per_key]))
