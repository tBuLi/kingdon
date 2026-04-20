""" Tests for the array syntax of kingdon. """
import random
import numpy as np
import pytest
from einops import rearrange, reduce, pack, unpack, einsum
from einops.tests import FLOAT_REDUCTIONS
from einops.tests.test_layers import rearrangement_patterns, reduction_patterns

from kingdon import Algebra, MultiVector, stack
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

def test_pack_unpack():
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

test_pack_unpack_config = [
    (np.random.randn, np.ndarray, '*', (2, 3)),
    (lambda n: np.random.randn(n, 4), np.ndarray, '* n', (2, 3, 4)),
    (lambda n: np.random.randn(n, 4), np.ndarray, 'n *', (2, 4, 3)),
]
@pytest.mark.parametrize("randn, expected_type, pattern, expected_shape", test_pack_unpack_config)
def test_pack_unpack_list(randn, expected_type, pattern, expected_shape):
    """Test pack and unpack with a list of multivectors and different patterns."""
    mvs = [alg.vector(randn(2)) for _ in range(3)]
    packed, ps = pack(mvs, pattern)
    assert packed.shape == expected_shape
    assert isinstance(packed, MultiVector)
    assert isinstance(packed.values(), expected_type)
    unpacked = unpack(packed, ps, pattern)
    assert isinstance(unpacked, list)
    assert len(unpacked) == len(mvs)
    for x, y in zip(unpacked, mvs):
        assert x.shape == y.shape
        assert np.allclose(x.values(), y.values())

test_stack_config = [
    (np.random.randn, np.ndarray, np.stack, (2, 3)),
    (lambda n: [random.gauss() for _ in range(n)], list, list, (2, 3)),
    (lambda n: np.random.randn(n, 4), np.ndarray, np.stack, (2, 3, 4)),
    (lambda n: [[random.gauss() for _ in range(4)] for _ in range(n)], list, list, (2, 3, 4)),
]
@pytest.mark.parametrize("randn, expected_type, stack_func, expected_shape", test_stack_config)
def test_stack(randn, expected_type, stack_func, expected_shape):
    """Test kingdon.stack with a list of multivectors and different stack functions."""
    mvs = [alg.vector(randn(2)) for _ in range(3)]
    stacked = stack(mvs, stack_func=stack_func)
    assert stacked.shape == expected_shape
    assert isinstance(stacked, MultiVector)
    assert isinstance(stacked.values(), expected_type)
    assert np.allclose(stacked[0].values(), mvs[0].values())
    assert np.allclose(stacked[1].values(), mvs[1].values())
    assert np.allclose(stacked[2].values(), mvs[2].values())
    different_mv = alg.evenmv(randn(2))
    with pytest.raises(TypeError, match='keys'):
        stack([*mvs, different_mv])
    with pytest.raises(TypeError, match='shape'):
        stack([stacked, alg.vector(randn(2))])
