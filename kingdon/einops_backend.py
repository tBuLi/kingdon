from __future__ import annotations

import string
from functools import lru_cache

from einops._backends import AbstractBackend, get_backend

from kingdon.multivector import MultiVector, _coefficients, _union_keys


@lru_cache(256)
def _add_blade_axis(pattern: str, is_mv: tuple[bool, ...]) -> str:
    """
    Add the (hidden) blade axis to a compact einsum `pattern`.

    By the time :func:`einops.einsum` calls into a backend the pattern has been
    compactified into single letter axis names, e.g. :code:`'i, i j -> j'` becomes
    :code:`'a,ab->b'`. Since :code:`mv.shape` does not expose the blade axis, the
    values of a multivector have one axis more than its pattern accounts for.
    So we prepend an unused letter to every multivector operand and to the output,
    turning :code:`'a,ab->b'` into :code:`'ca,ab->cb'`.

    :param pattern: compact einsum pattern, as produced by einops.
    :param is_mv: for each operand, whether it is a :class:`~kingdon.multivector.MultiVector`.
    """
    lefts, _, right = pattern.partition('->')
    lefts = lefts.split(',')
    if len(lefts) != len(is_mv):
        raise ValueError(f'The pattern {pattern} does not match the number of operands.')
    blade = next((letter for letter in string.ascii_letters if letter not in pattern), None)
    if blade is None:
        raise RuntimeError('Too many axes in einsum: no axis name left for the blade axis.')

    lefts = [f'{blade}{left}' if mv else left for left, mv in zip(lefts, is_mv)]
    return f"{','.join(lefts)}->{blade if any(is_mv) else ''}{right}"


def _zeros_like(x):
    """
    Zeros with the same shape, dtype and device as the coefficient `x`.

    The einops backends offer no way to create a tensor, so we make one out of `x` itself:
    give it an axis of length zero and then sum over that axis. Summing nothing is exactly
    zero, and since the values of `x` are never read this holds even if they are :code:`NaN`.
    Unlike :func:`~kingdon.multivector._zeros_like` this goes through the einops primitives,
    which also reach the array types that do not expose the python array API.
    """
    backend = get_backend(x)
    empty = backend.tile(backend.add_axis(x, 0), (0, *(1,) * len(backend.shape(x))))
    return backend.reduce(empty, 'sum', (0,))


class KingdonBackend(AbstractBackend):
    framework_name = "kingdon"

    def is_appropriate_type(self, x: object) -> bool:
        return isinstance(x, MultiVector)

    def shape(self, x: MultiVector) -> tuple[int, ...]:
        return x.shape

    def reshape(self, x: MultiVector, shape: tuple[int, ...]) -> MultiVector:
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.reshape(v, shape) for v in values]
        else:
            values = get_backend(values).reshape(values, (len(x._keys), *shape))
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def transpose(self, x: MultiVector, axes: tuple[int, ...]) -> MultiVector:
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.transpose(v, axes) for v in values]
        else:
            values = get_backend(values).transpose(values, (0, *(i + 1 for i in axes)))
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def reduce(self, x: MultiVector, operation: str, axes: tuple[int, ...]) -> MultiVector:
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.reduce(v, operation, axes) for v in values]
        else:
            # The axes have to be a tuple; numpy interprets a list as a single axis.
            values = get_backend(values).reduce(values, operation, tuple(i + 1 for i in axes))
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def add_axis(self, x: MultiVector, new_position: int) -> MultiVector:
        """ Insert an axis of length one at `new_position`, used by :func:`einops.repeat`. """
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.add_axis(v, new_position) for v in values]
        else:
            values = get_backend(values).add_axis(values, new_position + 1)
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def tile(self, x: MultiVector, repeats: tuple[int, ...]) -> MultiVector:
        """ Repeat `x` `repeats` times along each of its axes, used by :func:`einops.repeat`. """
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.tile(v, repeats) for v in values]
        else:
            values = get_backend(values).tile(values, (1, *repeats))
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def to_numpy(self, x: MultiVector) -> MultiVector:
        """
        Convert the coefficients of `x` to :code:`numpy` arrays.

        Unlike the other backends this returns a multivector, not an :code:`numpy.ndarray`,
        since a multivector is a container of tensors rather than a tensor itself.
        Use :code:`mv.values()` on the result to get at the raw array.
        """
        values = x._values
        if isinstance(values, (list, tuple)):
            backend = get_backend(values[0])
            values = [backend.to_numpy(v) for v in values]
        else:
            values = get_backend(values).to_numpy(values)
        return x.fromkeysvalues(x.algebra, x._keys, values)

    def concat(self, mvs: list[MultiVector], axis: int) -> MultiVector:
        """
        Concatenate `mvs` along `axis`, used by :func:`einops.pack`.

        All multivectors must be of the same type, since it is the type that gives the blades
        their meaning, but they need not have the same keys: the result has the union of the
        keys of `mvs`, and a blade that an input does not have contributes zeros. So a sparse
        input is densified to the keys of the result, e.g. packing an :code:`alg.vector(e1=...)`
        with an :code:`alg.vector(...)` gives the former an explicit zero on :code:`e2`.
        """
        if len({type(mv) for mv in mvs}) != 1:
            raise TypeError('To concat all multivectors must have the same type.')
        keys = _union_keys(mvs)
        # The result holds its coefficients in a single array only if all the inputs do.
        asarray = all(not isinstance(mv._values, (list, tuple)) for mv in mvs)

        if asarray and all(mv._keys == keys for mv in mvs):
            values = get_backend(mvs[0]._values).concat([mv._values for mv in mvs], axis + 1)
        else:
            coefficients = [_coefficients(mv, keys, _zeros_like) for mv in mvs]
            backend = get_backend(coefficients[0][0])
            values = [backend.concat([c[i] for c in coefficients], axis) for i in range(len(keys))]
            if asarray:
                values = backend.stack_on_zeroth_dimension(values)
        return mvs[0].fromkeysvalues(mvs[0].algebra, keys, values)

    def einsum(self, pattern, *operands):
        """
        Einstein summation over the array dimensions of multivectors, e.g.
        :code:`einsum(vec, weight, 'i, i j -> j')`. The blade axis is implicit: patterns
        describe :code:`mv.shape` only, and the summation is performed blade by blade.
        Consequently all multivector operands must have the same keys, with the exception
        of scalars, which multiply every blade.
        """
        mvs = [op for op in operands if isinstance(op, MultiVector)]
        base = next((mv for mv in mvs if mv._keys != (0,)), mvs[0])
        keys = base._keys

        is_mv, values = [], []
        for op in operands:
            if not isinstance(op, MultiVector):
                is_mv.append(False)
                values.append(op)
            elif op._keys == keys:
                is_mv.append(True)
                values.append(self._as_tensor(op))
            elif op._keys == (0,):
                # A scalar has no blade axis to speak of, so it enters as a plain tensor.
                is_mv.append(False)
                values.append(op._values[0])
            else:
                raise TypeError('To einsum all multivectors must have the same keys (i.e. basis blades), '
                                'with the exception of scalars.')

        backend = get_backend(next(v for v, mv in zip(values, is_mv) if mv))
        values = backend.einsum(_add_blade_axis(pattern, tuple(is_mv)), *values)
        return base.fromkeysvalues(base.algebra, keys, values)

    @staticmethod
    def _as_tensor(mv: MultiVector):
        """ The values of `mv` as a single tensor whose first axis is the blade axis. """
        values = mv._values
        if isinstance(values, (list, tuple)):
            return get_backend(values[0]).stack_on_zeroth_dimension(values)
        return values

    def is_float_type(self, x: MultiVector) -> bool:
        values = x._values
        first = values[0] if isinstance(values, (list, tuple)) else values
        # Ask the coefficients' own backend, since e.g. torch dtypes are not strings.
        return hasattr(first, "dtype") and get_backend(first).is_float_type(first)
