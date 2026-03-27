from __future__ import annotations

import numpy as np
from einops._backends import AbstractBackend

from kingdon.multivector import MultiVector


class KingdonBackend(AbstractBackend):
    framework_name = "kingdon"

    def is_appropriate_type(self, tensor: object) -> bool:
        return isinstance(tensor, MultiVector)

    def shape(self, x: MultiVector) -> tuple[int, ...]:
        # Expose only the non-blade (batch) dimensions to einops.
        return x.shape[1:]

    def reshape(self, x: MultiVector, shape: tuple[int, ...]) -> MultiVector:
        return x.map(lambda v: v.reshape(shape))

    def transpose(self, x: MultiVector, axes: tuple[int, ...]) -> MultiVector:
        return x.map(lambda v: v.transpose(axes))

    def reduce(self, x: MultiVector, operation: str, axes: tuple[int, ...]) -> MultiVector:
        return x.map(lambda v: getattr(v, operation)(axis=axes))

    def add_axis(self, x: MultiVector, new_position: int) -> MultiVector:
        return x.map(lambda v: np.expand_dims(v, new_position))

    def stack_on_zeroth_dimension(self, tensors: list[MultiVector]) -> MultiVector:
        keys = tensors[0].keys()
        values = [np.stack([t._values[i] for t in tensors]) for i in range(len(keys))]
        return MultiVector.fromkeysvalues(tensors[0].algebra, keys, values)

    def tile(self, x: MultiVector, repeats: tuple[int, ...]) -> MultiVector:
        return x.map(lambda v: np.tile(v, repeats))

    def concat(self, tensors: list[MultiVector], axis: int) -> MultiVector:
        keys = tensors[0].keys()
        values = [np.concatenate([t._values[i] for t in tensors], axis=axis) for i in range(len(keys))]
        return MultiVector.fromkeysvalues(tensors[0].algebra, keys, values)

    def is_float_type(self, x: MultiVector) -> bool:
        first_val = list(x.values())[0]
        return hasattr(first_val, "dtype") and first_val.dtype in (
            "float16", "float32", "float64", "float128", "bfloat16",
        )
