""" Tests for the array syntax of kingdon. """

import numpy as np
import pytest
from einops import rearrange, reduce
from einops.tests import FLOAT_REDUCTIONS
from einops.tests.test_layers import rearrangement_patterns, reduction_patterns

from kingdon import Algebra
from kingdon.multivector import MultiVector
import kingdon.einops_backend  # registers KingdonBackend with einops

# A small algebra used throughout these tests.
alg = Algebra(2)  # 4 blades: e, e1, e2, e12
KEYS = tuple(alg.canon2bin.values())


def make_mv(input_shape):
    """Create a full multivector whose coefficients are distinct float32 arrays of shape (len(KEYS), *input_shape)."""
    shape = (len(KEYS), *input_shape)
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    return MultiVector.fromkeysvalues(alg, KEYS, values)


@pytest.mark.parametrize("pattern", rearrangement_patterns)
def test_rearrangement(pattern):
    """Rearranging a multivector applies the same rearrangement to every coefficient."""
    x = make_mv(pattern.input_shape)
    y = rearrange(x, pattern.pattern, **pattern.axes_lengths)
    y_alt = x.map(lambda v: rearrange(v, pattern.pattern, **pattern.axes_lengths))

    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == y_alt.shape
    assert np.allclose(y.values(), y_alt.values())


@pytest.mark.parametrize("reduction", FLOAT_REDUCTIONS)
@pytest.mark.parametrize("pattern", reduction_patterns)
def test_reduction(pattern, reduction):
    """Reducing a multivector applies the same reduction to every coefficient."""
    x = make_mv(pattern.input_shape).map(lambda v: v / v.mean())
    y = reduce(x, pattern.pattern, reduction, **pattern.axes_lengths)
    y_alt = x.map(lambda v: reduce(v, pattern.pattern, reduction, **pattern.axes_lengths))

    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == y_alt.shape  # Using einops on the mv should be the same as component-wise einops.
    assert np.allclose(y.values(), y_alt.values())
