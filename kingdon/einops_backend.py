from __future__ import annotations

import numpy as np
from einops._backends import AbstractBackend, get_backend

from kingdon.multivector import MultiVector


class KingdonBackend(AbstractBackend):
    framework_name = "kingdon"

    def is_appropriate_type(self, tensor: object) -> bool:
        return isinstance(tensor, MultiVector)

    def shape(self, x: MultiVector) -> tuple[int, ...]:
        # Expose only the non-blade (batch) dimensions to einops.
        return x.shape[1:]

    def reshape(self, x: MultiVector, shape: tuple[int, ...]) -> MultiVector:
        if isinstance(x._values, (list, tuple)):
            return x.map(lambda v: get_backend(v).reshape(v, shape))
        values = get_backend(x._values).reshape(x._values, (len(x._values), *shape))
        return MultiVector(x.algebra, keys=x.keys(), values=values)

    def transpose(self, x: MultiVector, axes: tuple[int, ...]) -> MultiVector:
        if isinstance(x._values, (list, tuple)):
            return x.map(lambda v: get_backend(v).transpose(v, axes))
        axes = [0, *(i + 1 for i in axes)]
        values = get_backend(x._values).transpose(x._values, axes)
        return MultiVector(x.algebra, keys=x.keys(), values=values)

    def reduce(self, x: MultiVector, operation: str, axes: tuple[int, ...]) -> MultiVector:
        if isinstance(x._values, (list, tuple)):
            return x.map(lambda v: get_backend(v).reduce(v, operation, axes))
        axes = [0, *(i + 1 for i in axes)]
        values = get_backend(x._values).reduce(x._values, operation, axes)
        return MultiVector(x.algebra, keys=x.keys(), values=values)

    def concat(self, mvs: list[MultiVector], axis: int) -> MultiVector:
        keys = mvs[0].keys()
        if not all(mv.keys() == keys for mv in mvs[1:]):
            raise TypeError('To concat all multivectors must have the same keys (i.e. basis blades).')
        backend = get_backend(mvs[0].values())
        values = backend.concat([mv.values() for mv in mvs], axis + 1)
        return MultiVector.fromkeysvalues(mvs[0].algebra, keys, values)

    def einsum(self, pattern, *mvs: list[MultiVector]):
        base_mv = max([mv for mv in mvs if isinstance(mv, MultiVector)], key=lambda mv: len(mv.shape))
        _values = [mv._values if isinstance(mv, MultiVector) else mv for mv in mvs]
        backend = get_backend(_values[0])
        values = backend.einsum(pattern, *_values)
        return MultiVector(base_mv.algebra, keys=base_mv.keys(), values=values)

    def is_float_type(self, x: MultiVector) -> bool:
        first_val = list(x.values())[0]
        return hasattr(first_val, "dtype") and first_val.dtype in (
            "float16", "float32", "float64", "float128", "bfloat16",
        )
