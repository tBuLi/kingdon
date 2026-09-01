""" Tests for the array syntax of kingdon. """
import copy
import random

import einops
import numpy as np
import pytest
from einops import rearrange, reduce, repeat, pack, unpack, einsum, asnumpy, parse_shape
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
    """Reducing a multivector whose coefficients are a list of arrays reduces every coefficient."""
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


@pytest.mark.parametrize("reduction", FLOAT_REDUCTIONS)
def test_reduction_array_values(reduction):
    """Reducing a multivector whose coefficients are one stacked array reduces every coefficient,
    leaving the blade dimension of that array intact."""
    x = make_mv((3, 4, 5)).map(lambda v: v / v.mean())
    x_arr = MultiVector.fromkeysvalues(alg, x.keys(), np.stack(x.values()))
    y = reduce(x_arr, 'a b c -> a c', reduction)
    y_gt = reduce(x, 'a b c -> a c', reduction)

    assert isinstance(y.values(), np.ndarray)  # The blade dimension survives the reduction.
    assert y.values().shape == (len(KEYS), 3, 5)
    assert y.shape == y_gt.shape == (3, 5)
    assert np.allclose(y.values(), y_gt.values())


def test_repeat():
    """ Repeating a multivector applies the same repeat to every coefficient. """
    x = make_mv((3, 4))
    x_alt = x.map(lambda v: v)
    y = repeat(x, 'a b -> a b c', c=5)
    y_alt = repeat(x_alt, 'a b -> a b c', c=5)
    y_gt = x.map(lambda v: repeat(v, 'a b -> a b c', c=5))

    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == (3, 4, 5) == y_gt.shape
    assert np.allclose(y.values(), y_gt.values())
    assert type(x.values()) == type(y.values())

    assert y_alt.shape == y_gt.shape
    assert np.allclose(y_alt.values(), y_gt.values())
    assert type(x_alt.values()) == type(y_alt.values())

    # Repeating along an existing axis works just as well.
    z = repeat(x, 'a b -> (rep a) b', rep=2)
    assert z.shape == (6, 4)
    assert np.allclose(z.values(), x.map(lambda v: repeat(v, 'a b -> (rep a) b', rep=2)).values())

    # The type of the multivector is preserved.
    vec = alg.vector(np.random.randn(2, 3))
    assert type(repeat(vec, 'a -> a b', b=2)) is type(vec)


def test_asnumpy():
    """ asnumpy returns a multivector whose coefficients are numpy arrays. """
    x = make_mv((3, 4))
    y = asnumpy(x)
    assert isinstance(y, MultiVector)  # A multivector is a container of tensors, not a tensor itself.
    assert y.keys() == x.keys()
    assert y.shape == x.shape
    assert isinstance(y.values(), np.ndarray)
    assert np.allclose(y.values(), x.values())

    y_alt = asnumpy(x.map(lambda v: v))
    assert y_alt.shape == x.shape
    assert all(isinstance(v, np.ndarray) for v in y_alt.values())


def test_parse_shape():
    """ parse_shape only sees the array dimensions, not the blades. """
    x = make_mv((3, 4))
    assert parse_shape(x, 'a b') == {'a': 3, 'b': 4}
    assert parse_shape(x, 'a _') == {'a': 3}


def test_pack_unpack():
    inputs = [np.zeros([2, 3, 5]), np.zeros([2, 3, 7, 5]), np.zeros([2, 3, 7, 9, 5])]
    mvs = [alg.vector(i) for i in inputs]
    assert [mv.shape for mv in mvs] == [(3, 5), (3, 7, 5), (3, 7, 9, 5)]
    packed, ps = pack(mvs, 'i * k')  # i matches 3, k matches 5.
    assert packed.shape == (3, 71, 5)
    assert type(packed.values()) == np.ndarray
    inputs_unpacked = unpack(packed, ps, 'i * k')
    assert [x.shape for x in inputs_unpacked] == [mv.shape for mv in mvs]


def test_pack_unpack_list_values():
    """ pack & unpack also work when the coefficients are a list of arrays. """
    mvs = [alg.vector(np.zeros([2, 3, 5])).map(lambda v: v),
           alg.vector(np.zeros([2, 3, 7, 5])).map(lambda v: v)]
    packed, ps = pack(mvs, 'i * k')
    assert packed.shape == (3, 8, 5)
    assert isinstance(packed.values(), list)
    unpacked = unpack(packed, ps, 'i * k')
    assert [x.shape for x in unpacked] == [mv.shape for mv in mvs]

def test_pack_unpack_inhomogenous():
    """ Pack should work for inhomogenous mvs so long as they are the same type. """
    mvs = [alg.vector(np.ones([2, 3, 5])).map(lambda v: v),
           alg.vector(e1=np.ones([3, 7, 5]))]
    packed, ps = pack(mvs, 'i * k')
    assert packed.shape == (3, 8, 5)
    assert isinstance(packed.values(), list)
    unpacked = unpack(packed, ps, 'i * k')
    assert [x.shape for x in unpacked] == [mv.shape for mv in mvs]

    # The keys are relaxed to those of the mv with the most keys, the mv that had all of them
    # comes back unharmed, and the blade the other one lacked is zero.
    assert type(packed) is type(mvs[0]) and packed.keys() == mvs[0].keys()
    assert all(x.keys() == packed.keys() for x in unpacked)
    assert np.allclose(unpacked[0].values(), mvs[0].values())
    assert np.allclose(unpacked[1].e1, mvs[1].e1) and not unpacked[1].e2.any()

    # Those zeros are made from the coefficients of the mv that is missing a blade, so they
    # follow its dtype and are unaffected by any NaNs it happens to carry.
    nan = alg.vector(e1=np.full([3, 7, 5], np.nan, dtype='float32'))
    packed, ps = pack([mvs[0].map(lambda v: v.astype('float32')), nan], 'i * k')
    assert all(v.dtype == np.dtype('float32') for v in packed.values())
    assert not unpack(packed, ps, 'i * k')[1].e2.any()


def test_pack_unpack_inhomogenous_asarray():
    """ Inhomogenous mvs that all hold their coefficients as one array give a result that does too. """
    sparse = alg.vector(e1=np.ones([3, 7, 5]))
    sparse = type(sparse).fromkeysvalues(alg, sparse.keys(), np.stack(sparse.values()))
    packed, ps = pack([alg.vector(np.ones([2, 3, 5])), sparse], 'i * k')
    assert isinstance(packed.values(), np.ndarray)
    assert packed.values().shape == (2, 3, 8, 5)  # The blade axis survives the zero filling.
    assert not unpack(packed, ps, 'i * k')[1].e2.any()


def test_pack_different_types():
    """ Relaxing the keys is only allowed within one type, since the type fixes their meaning. """
    with pytest.raises(TypeError):
        pack([alg.vector(np.ones([2, 3, 5])), alg.bivector(np.ones([1, 3, 5]))], 'i * k')

def test_einsum():
    vec_of_matrices = np.random.randn(2, 10, 10)
    vec = alg.vector(vec_of_matrices)
    trace = einsum(vec, "i i ->")
    assert trace.shape == ()
    assert isinstance(trace, MultiVector)
    assert isinstance(trace.values(), np.ndarray)
    assert np.allclose(trace.values(), np.einsum("zii->z", vec_of_matrices))

    weight = np.random.randn(10, 20)
    matmul = einsum(vec, weight, "i j, j k -> i k")
    assert matmul.shape == (10, 20)
    assert isinstance(matmul, MultiVector)
    assert isinstance(matmul.values(), np.ndarray)
    assert np.allclose(matmul.values(), np.einsum("zij,jk->zik", vec_of_matrices, weight))

    # Patterns which spell out the batch dimensions with an ellipsis keep working,
    # the ellipsis simply no longer has to swallow the blade axis.
    assert np.allclose(einsum(vec, "... i i -> ...").values(), trace.values())
    assert np.allclose(einsum(vec, weight, "... j, j k -> ... k").values(), matmul.values())


def test_einsum_types():
    """ The type of the multivector is preserved, and list-valued multivectors are supported. """
    vec = alg.vector(np.random.randn(2, 10))
    weight = np.random.randn(10, 20)
    matmul = einsum(vec, weight, "i, i j -> j")
    assert type(matmul) is type(vec)
    assert matmul.keys() == vec.keys()
    assert matmul.shape == (20,)

    vec_alt = vec.map(lambda v: v)  # values are a list of arrays instead of one array.
    matmul_alt = einsum(vec_alt, weight, "i, i j -> j")
    assert matmul_alt.shape == (20,)
    assert np.allclose(matmul_alt.values(), matmul.values())


def test_einsum_multiple_multivectors():
    """ Multivectors are combined blade by blade, and scalars multiply every blade. """
    x = alg.vector(np.random.randn(2, 10))
    y = alg.vector(np.random.randn(2, 10))
    prod = einsum(x, y, "i, i ->")
    assert prod.shape == ()
    assert prod.keys() == x.keys()
    assert np.allclose(prod.values(), np.einsum("zi,zi->z", x.values(), y.values()))

    s = alg.scalar(np.random.randn(1, 10))
    scaled = einsum(x, s, "i, i -> i")
    assert scaled.keys() == x.keys()
    assert np.allclose(scaled.values(), x.values() * s.values())

    # Scalars are also fine as the leading operand.
    assert np.allclose(einsum(s, x, "i, i -> i").values(), scaled.values())

    with pytest.raises(TypeError):
        einsum(x, alg.bivector(np.random.randn(1, 10)), "i, i -> i")

test_pack_unpack_config = [
    (np.random.randn, np.ndarray, '*', (3,)),
    (lambda n: np.random.randn(n, 4), np.ndarray, '* n', (3, 4)),
    (lambda n: np.random.randn(n, 4), np.ndarray, 'n *', (4, 3)),
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
    (np.random.randn, np.ndarray, np.stack, (3,)),
    (lambda n: [random.gauss() for _ in range(n)], list, list, (3,)),
    (lambda n: np.random.randn(n, 4), np.ndarray, np.stack, (3, 4)),
    (lambda n: [[random.gauss() for _ in range(4)] for _ in range(n)], list, list, (3, 4)),
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
    with pytest.raises(TypeError, match='type'):
        stack([*mvs, different_mv])
    with pytest.raises(TypeError, match='shape'):
        stack([stacked, alg.vector(randn(2))])

    # Within one type the keys are relaxed to their union, where the blades an input does
    # not have contribute zeros.
    sparse = alg.vector(e1=randn(2)[0])
    relaxed = stack([*mvs, sparse], stack_func=stack_func)
    assert relaxed.keys() == mvs[0].keys()
    assert relaxed.shape == (len(mvs) + 1, *expected_shape[1:])  # The zeros are shaped correctly.
    assert np.allclose(relaxed[len(mvs)].e1, sparse.e1)
    assert not np.any(relaxed[len(mvs)].e2)

def test_asarray_algebra():
    # Use np.array everywhere
    alg = Algebra(2, values_asarray=np.asarray)
    x = alg.vector(e1=2, e2=3)
    y = alg.vector([4, 5])
    assert isinstance(x.values(), np.ndarray)
    assert isinstance(y.values(), np.ndarray)
    z = x * y
    assert isinstance(z.values(), np.ndarray)

    # Use np.ndarray only for custom operator
    alg = Algebra(2)
    x = alg.vector(e1=2, e2=3)
    y = alg.vector([4, 5])
    assert isinstance(x.values(), list)
    assert isinstance(y.values(), list)
    z = x * y
    assert isinstance(z.values(), list)

    @alg.jit(values_asarray=np.asarray)
    def my_func(x, y):
        return x + y
    w = my_func(x, y)
    assert isinstance(w.values(), np.ndarray)

    # Inhomogenous data
    x = alg.vector(np.array([[1., 2.], [3., 4.]]))
    s = alg.scalar(np.array([1.0]))
    with pytest.raises(ValueError):
        my_func(x, s)

    @alg.jit(values_asarray=lambda values: np.asarray(np.broadcast_arrays(*values)))
    def my_broadcasted_func(x, y):
        return x + y

    w = my_broadcasted_func(x, s)
    assert isinstance(w.values(), np.ndarray)
    assert w.shape == (2,)


def test_asarray_raw():
    """ values_asarray is applied to numeric values, but symbolic values are left alone. """
    def asarray(values):
        return np.asarray(np.broadcast_arrays(*values))
    alg = Algebra(2, values_asarray=asarray)

    # Numeric multivectors are cast to arrays, wherever they are created.
    assert isinstance(alg.vector([1., 2.]).values(), np.ndarray)
    assert isinstance((alg.vector([1., 2.]) * alg.vector([3., 4.])).values(), np.ndarray)
    assert isinstance(alg.blades.e1.values(), np.ndarray)

    # Symbolic multivectors keep their values as a list, because casting sympy
    # expressions to an array would give an array of dtype object.
    x = alg.multivector(name='x')
    assert x.issymbolic
    assert isinstance(x.values(), list)

    # Copies are exact replicas, so they keep the values as they are.
    assert copy.copy(x).values() is x.values()
    assert isinstance(copy.deepcopy(x).values(), list)
    assert copy.deepcopy(x).values() == x.values()

    # Multivectors without any values are created without complaint.
    assert len(alg.vector([1., 2.]).filter(lambda v: False).values()) == 0


def test_136():
    """ Multivectors should be compatible with numpy.broadcast. """
    pga = Algebra.fromname("3DPGA")
    nb_points = 10
    xyz = np.random.rand(3, nb_points)
    point = (pga.evector(xyz) + pga.blades.e0).dual()
    vector = pga.evector(e1=4.0, e2=8.0, e3=-1.2)
    some_object = [None] * nb_points

    with pytest.raises(TypeError, match='0-dimensional'):
        len(vector)
    with pytest.raises(TypeError, match='0-dimensional'):
        iter(vector)

    fo = np.broadcast(5.0, some_object)
    assert fo.shape == (10,)
    for f, o in fo:
        print(f"float: {f}, object: {o}")  # broadcasts normally: prints 10x "float: 5.0, object: None"

    pv = np.broadcast(point, vector)
    assert pv.shape == (10,)
    for p, v in pv:  # empty iterable, shape = (10, 0)
        print(f"point: {p}, vector: {v}")

    assert point.shape == (10,)
    po = np.broadcast(point, some_object)
    assert po.shape == (10,)
    for p, o in po:  # ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (10, 0) and arg 1 with shape (10,).
        print(f"point: {p}, object: {o}")

def test_newaxis_vs_stack():
    def asarray(values):
        return np.asarray(np.broadcast_arrays(*values))

    pga = Algebra.fromname("3DPGA", values_asarray=asarray)
    N, M = 4, 5
    point_vals = np.random.rand(N, 3)
    line_vals = np.random.rand(M, 6)
    points = pga.point(einops.rearrange(point_vals, '... blades -> blades ...'))
    bivectors = pga.bivector(einops.rearrange(line_vals, '... blades -> blades ...'))
    lines = bivectors.normalized()
    # Check if the broadcasting results in the same as brute force.
    projected_points = points[:, None] @ lines[None, :]
    assert not (projected_points - stack([p @ lines for p in points])).values().any()  # Not any value should be true-ish.
    # Check if the values are just a view of the original arrays.
    assert points.values().base is point_vals
    assert bivectors.values().base is line_vals
