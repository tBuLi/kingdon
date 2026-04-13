""" Tests for the array syntax of kingdon. """

import numpy as np
import pytest
from einops import rearrange, reduce, pack, unpack, einsum
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
    x_alt = x.map(lambda v: v)
    y = rearrange(x, pattern.pattern, **pattern.axes_lengths)
    y_alt = rearrange(x_alt, pattern.pattern, **pattern.axes_lengths)
    y_gt = x.map(lambda v: rearrange(v, pattern.pattern, **pattern.axes_lengths))

    # einops should preserve type of .values() and match ground truth (gt)
    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == y_gt.shape
    assert np.allclose(y.values(), y_gt.values())
    assert type(x.values()) == type(y.values())

    # So for the alt construction the types should be a list instead of np.ndarray
    assert isinstance(y_alt, MultiVector)
    assert y_alt.keys() == x_alt.keys()
    assert y_alt.shape == y_gt.shape
    assert np.allclose(y_alt.values(), y_gt.values())
    assert type(x_alt.values()) == type(y_alt.values())


@pytest.mark.parametrize("reduction", FLOAT_REDUCTIONS)
@pytest.mark.parametrize("pattern", reduction_patterns)
def test_reduction(pattern, reduction):
    """Reducing a multivector applies the same reduction to every coefficient."""
    x = make_mv(pattern.input_shape).map(lambda v: v / v.mean())
    x_alt = x.map(lambda v: v)
    y = reduce(x, pattern.pattern, reduction, **pattern.axes_lengths)
    y_alt = reduce(x_alt, pattern.pattern, reduction, **pattern.axes_lengths)
    y_gt = x.map(lambda v: reduce(v, pattern.pattern, reduction, **pattern.axes_lengths))

    # einops should preserve type of .values() and match ground truth (gt)
    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == y_alt.shape  # Using einops on the mv should be the same as component-wise einops.
    assert np.allclose(y.values(), y_alt.values())
    assert type(x.values()) == type(y.values())

    # So for the alt construction the types should be a list instead of np.ndarray
    assert isinstance(y_alt, MultiVector)
    assert y_alt.keys() == x_alt.keys()
    assert y_alt.shape == y_gt.shape
    assert np.allclose(y_alt.values(), y_gt.values())
    assert type(x_alt.values()) == type(y_alt.values())

def test_pack():
    inputs = [np.zeros([2, 3, 5]), np.zeros([2, 3, 7, 5]), np.zeros([2, 3, 7, 9, 5])]
    mvs = [alg.vector(i) for i in inputs]
    packed, ps = pack(mvs, 'j * k')
    assert packed.shape == (2, 3, 71, 5)
    assert type(packed.values()) == np.ndarray
    inputs_unpacked = unpack(packed, ps, 'j * k')
    assert [x.shape for x in inputs_unpacked] == [x.shape for x in mvs]

def test_einsum():
    vec_of_matrices = np.random.randn(2, 10, 10)
    vec = alg.vector(vec_of_matrices)
    trace = einsum(vec, "... i i -> ...")
    assert trace.shape == (2,)
    assert isinstance(trace, MultiVector)
    assert isinstance(trace.values(), np.ndarray)

    weight = np.random.randn(10, 20)
    matmul = einsum(vec, weight, "... i, i j -> ... j")
    assert matmul.shape == (2, 10, 20)
    assert isinstance(matmul, MultiVector)
    assert isinstance(matmul.values(), np.ndarray)
