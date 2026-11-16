""" Tests for the array syntax of kingdon. """
import copy
import math
from functools import partial

import einops
import numpy as np
import pytest
from array_api_compat import (array_namespace, is_array_api_obj, is_cupy_array,
                              is_numpy_namespace, is_torch_namespace)
from einops import rearrange, reduce, repeat, pack, unpack, einsum, asnumpy, parse_shape
from einops.tests import FLOAT_REDUCTIONS
from einops.tests.test_layers import rearrangement_patterns, reduction_patterns

from kingdon import Algebra, MultiVector, stack
import kingdon.einops_backend  # noqa: F401  # registers KingdonBackend with einops. Do not remove.

# A small algebra used throughout these tests.
alg = Algebra(2)  # 4 blades: e, e1, e2, e12
KEYS = tuple(alg.canon2bin.values())

# The array libraries these tests run against: those that einops has a backend for and that
# array-api-compat can wrap into an array API namespace. Any that is not installed is skipped.
ARRAY_LIBS = {'numpy': 'numpy', 'torch': 'torch', 'jax': 'jax.numpy', 'cupy': 'cupy'}

# Stacking arrays of different shapes into one array is an error, which numpy, jax and cupy
# spell as a ValueError but torch as a RuntimeError.
STACK_ERRORS = (ValueError, RuntimeError)


@pytest.fixture(params=list(ARRAY_LIBS))
def xp(request):
    """ The array API namespace to run a test with, skipped when its library is not installed. """
    lib = pytest.importorskip(ARRAY_LIBS[request.param])
    if request.param == 'jax':
        # Without x64 mode jax silently turns float64 into float32, which is too coarse for the tests that ask for float64.
        import jax
        jax.config.update('jax_enable_x64', True)
    return array_namespace(lib.empty(0))


def to_numpy(x):
    """ `x` as a numpy array, whether it is one array, a list of them, or plain python numbers. """
    if isinstance(x, (list, tuple)):
        return np.asarray([to_numpy(v) for v in x])
    return asnumpy(x) if is_cupy_array(x) else np.asarray(x)


def allclose(a, b):
    return np.allclose(to_numpy(a), to_numpy(b))


def assert_mv_allclose(a, b, atol=1e-8):
    """Compare coefficients after aligning blade layouts, including implicit constants."""
    assert a.shape == b.shape
    for value in (a - b).values():
        np.testing.assert_allclose(to_numpy(value), 0, atol=atol)


def randn(xp, *shape):
    """ A standard normal array. The array API has no randomness, so numpy makes the numbers. """
    return xp.asarray(np.random.randn(*shape))


def rand(xp, *shape):
    """ A uniform [0, 1) array. """
    return xp.asarray(np.random.rand(*shape))


def test_equality(xp):
    """
    Multivectors with array coefficients compare equal when every coefficient is exactly equal, giving a single bool.
    Lists of arrays compare like the stacked array, a blade missing on one side counts as zero, and different shapes or algebras are unequal.
    """
    values = xp.asarray([[1., 2.], [3., 4.]], dtype=xp.float64)
    x = alg.vector(values)
    assert (x == alg.vector(values + 0)) is True
    assert x == alg.vector(list(values))
    assert x != alg.vector(values + 1e-10)
    assert x != alg.vector(values[:, :1])
    assert x != Algebra(1, 1).vector(values)
    assert x != alg.vector(e1=values[0])
    assert alg.vector(e1=values[0], e2=0 * values[1]) == alg.vector(e1=values[0])
    assert x != 1


@pytest.mark.parametrize('signature', [(2, 0, 0), (1, 1, 0), (1, 0, 1)])
def test_batched_simple_exp(xp, signature):
    """
    A batch of bivectors that square to -1, +1 or 0 exponentiates to a batch of rotations, boosts or translations,
    including a zero angle and the angles of plus and minus a half turn.
    The logarithm gives the bivectors back, also after scaling the rotors.
    """
    algebra = Algebra(*signature)
    angles = xp.asarray([-0.8, 0., 0.4, math.pi, -math.pi], dtype=xp.float64)
    bivectors = algebra.bivector(angles[None, :])
    rotors = bivectors.exp()
    assert_mv_allclose(rotors.log(), bivectors)
    assert_mv_allclose((3 * rotors).log(), bivectors)
    assert rotors.shape == angles.shape
    assert allclose((rotors * ~rotors).e, xp.ones_like(angles))
    if signature[2]:
        scalar, bivector = xp.ones_like(angles), angles
    elif signature[1]:
        scalar, bivector = xp.cosh(angles), xp.sinh(angles)
    else:
        scalar, bivector = xp.cos(angles), xp.sin(angles)
        assert_mv_allclose(algebra.bivector([2 * math.pi]).exp(), algebra.blades.e)
    assert allclose(rotors.e, scalar)
    assert allclose(rotors.grade(2).values(), [bivector])


def test_exp_mixed_squares(xp):
    """
    In CGA bivectors can square to positive, negative, or zero, generating rotations, translations, boosts, or dilations.
    A single batch that mixes all of these exponentiates each element correctly, keeps a real dtype,
    and the logarithm gives back every bivector.
    """
    algebra = Algebra(4, 1)
    # e1 ^ (e4 + e5) squares to zero.
    bivectors = algebra.bivector(
        e15=xp.asarray([1., 0., 2., 0.], dtype=xp.float64),
        e12=xp.asarray([0., 1., 0., 0.], dtype=xp.float64),
        e14=xp.asarray([0., 0., 2., 0.], dtype=xp.float64),
        e45=xp.asarray([0., 0., 0., 1.], dtype=xp.float64),  # Dilation generator; e45**2 = +1.
    )
    assert allclose((bivectors * bivectors).e, [1., -1., 0., 1.])

    # A bivector with these blades is not always simple, which kingdon can not rule out without reading the coefficients.
    with pytest.warns(UserWarning, match='simple'):
        rotors = bivectors.exp()
        assert_mv_allclose(rotors.log(), bivectors)
        assert_mv_allclose((3 * rotors).log(), bivectors)
    with pytest.warns(UserWarning, match='grades'):
        assert_mv_allclose((rotors + algebra.multivector(e1234=0.)).log(), bivectors)
    assert rotors.shape == (4,)
    assert allclose(rotors.e, [math.cosh(1), math.cos(1), 1., math.cosh(1)])
    assert allclose(rotors.e15, [math.sinh(1), 0., 2., 0.])
    assert allclose(rotors.e12, [0., math.sin(1), 0., 0.])
    assert allclose(rotors.e14, [0., 0., 2., 0.])
    assert allclose(rotors.e45, [0., 0., 0., math.sinh(1)])
    assert all(xp.isdtype(v.dtype, 'real floating') for v in rotors.values())


def test_exp_log_can_be_traced():
    """ jax can compile exp and log, and map them over a batch. """
    jax = pytest.importorskip('jax')
    angles = jax.numpy.asarray([-0.8, 0., 0.4])
    for algebra, key in ((Algebra(2), 'e12'), (Algebra(1, 1), 'e12'), (Algebra(1, 0, 1), 'e01')):
        roundtrip = lambda x: getattr(algebra.bivector(**{key: x}).exp().log(), key)
        assert allclose(jax.jit(roundtrip)(angles), angles)
        assert allclose(jax.vmap(roundtrip)(angles), angles)


def test_torch_exp_log_gradients():
    """
    exp and log of torch tensors give the right gradients, also at a zero angle and at a large angle.
    """
    torch = pytest.importorskip('torch')
    angles = torch.tensor([-0.8, 0., 0.4, 800.], dtype=torch.float64, requires_grad=True)
    rotor = Algebra(2).bivector(angles[None, :]).exp()
    (rotor.e + rotor.e12).sum().backward()
    torch.testing.assert_close(angles.grad, torch.cos(angles) - torch.sin(angles))

    angles = torch.tensor([-0.8, 0., 0.4], dtype=torch.float64, requires_grad=True)
    Algebra(2).bivector(angles[None, :]).exp().log().e12.sum().backward()
    torch.testing.assert_close(angles.grad, torch.ones_like(angles))


def values_asarray(xp, values, broadcast=False):
    """
    The coefficients of a multivector as a single array, to pass as :code:`values_asarray`.

    This is what :func:`numpy.asarray` does for a list of arrays, but spelled in the array API,
    which only knows how to build an array out of other arrays by stacking them.

    :param broadcast: first broadcast the coefficients against each other, so that coefficients
        of different shapes still give one array.
    """
    values = [xp.asarray(v) for v in values]
    if not values:  # A multivector without coefficients has nothing to stack.
        return xp.asarray([])
    return xp.stack(xp.broadcast_arrays(*values) if broadcast else values)


def make_mv(xp, input_shape):
    """Create a full multivector whose coefficients are distinct float32 arrays of shape (len(KEYS), *input_shape)."""
    shape = (len(KEYS), *input_shape)
    values = xp.reshape(xp.arange(math.prod(shape), dtype=xp.float32), shape)
    return MultiVector.fromkeysvalues(alg, KEYS, values)


@pytest.mark.parametrize("pattern", rearrangement_patterns)
def test_rearrangement(xp, pattern):
    """Rearranging a multivector applies the same rearrangement to every coefficient."""
    x = make_mv(xp, pattern.input_shape)
    x_alt = x.map(lambda v: v)
    y = rearrange(x, pattern.pattern, **pattern.axes_lengths)
    y_alt = rearrange(x_alt, pattern.pattern, **pattern.axes_lengths)
    y_gt = x.map(lambda v: rearrange(v, pattern.pattern, **pattern.axes_lengths))

    # einops should preserve type of .values() and match ground truth (gt)
    assert type(y) is type(x) and y.keys() == x.keys()
    assert y == y_gt
    assert type(x.values()) == type(y.values())

    # So for the alt construction the types should be a list instead of an array
    assert type(y_alt) is type(x_alt) and y_alt.keys() == x_alt.keys()
    assert y_alt == y_gt
    assert type(x_alt.values()) == type(y_alt.values())


@pytest.mark.parametrize("reduction", FLOAT_REDUCTIONS)
@pytest.mark.parametrize("pattern", reduction_patterns)
def test_reduction(xp, pattern, reduction):
    """Reducing a multivector whose coefficients are a list of arrays reduces every coefficient."""
    x = make_mv(xp, pattern.input_shape).map(lambda v: v / xp.mean(v))
    x_alt = x.map(lambda v: v)
    y = reduce(x, pattern.pattern, reduction, **pattern.axes_lengths)
    y_alt = reduce(x_alt, pattern.pattern, reduction, **pattern.axes_lengths)
    y_gt = x.map(lambda v: reduce(v, pattern.pattern, reduction, **pattern.axes_lengths))

    # einops should preserve type of .values() and match ground truth (gt)
    assert isinstance(y, MultiVector)
    assert y.keys() == x.keys()
    assert y.shape == y_alt.shape  # Using einops on the mv should be the same as component-wise einops.
    assert allclose(y.values(), y_alt.values())
    assert type(x.values()) == type(y.values())

    # So for the alt construction the types should be a list instead of an array
    assert isinstance(y_alt, MultiVector)
    assert y_alt.keys() == x_alt.keys()
    assert y_alt.shape == y_gt.shape
    assert allclose(y_alt.values(), y_gt.values())
    assert type(x_alt.values()) == type(y_alt.values())


@pytest.mark.parametrize("reduction", FLOAT_REDUCTIONS)
def test_reduction_array_values(xp, reduction):
    """Reducing a multivector whose coefficients are one stacked array reduces every coefficient,
    leaving the blade dimension of that array intact."""
    x = make_mv(xp, (3, 4, 5)).map(lambda v: v / xp.mean(v))
    x_arr = MultiVector.fromkeysvalues(alg, x.keys(), xp.stack(list(x.values())))
    y = reduce(x_arr, 'a b c -> a c', reduction)
    y_gt = reduce(x, 'a b c -> a c', reduction)

    assert is_array_api_obj(y.values())  # The blade dimension survives the reduction.
    assert tuple(y.values().shape) == (len(KEYS), 3, 5)
    assert y.shape == y_gt.shape == (3, 5)
    assert allclose(y.values(), y_gt.values())


def test_repeat(xp):
    """ Repeating a multivector applies the same repeat to every coefficient. """
    x = make_mv(xp, (3, 4))
    x_alt = x.map(lambda v: v)
    y = repeat(x, 'a b -> a b c', c=5)
    y_alt = repeat(x_alt, 'a b -> a b c', c=5)
    y_gt = x.map(lambda v: repeat(v, 'a b -> a b c', c=5))

    assert y.shape == (3, 4, 5)
    assert type(y) is type(x) and y.keys() == x.keys()
    assert y == y_gt
    assert type(x.values()) == type(y.values())

    assert y_alt == y_gt
    assert type(x_alt.values()) == type(y_alt.values())

    # Repeating along an existing axis works just as well.
    z = repeat(x, 'a b -> (rep a) b', rep=2)
    assert z.shape == (6, 4)
    assert z == x.map(lambda v: repeat(v, 'a b -> (rep a) b', rep=2))

    # The type of the multivector is preserved.
    vec = alg.vector(randn(xp, 2, 3))
    assert type(repeat(vec, 'a -> a b', b=2)) is type(vec)


def test_asnumpy(xp):
    """ asnumpy returns a multivector whose coefficients are numpy arrays. """
    x = make_mv(xp, (3, 4))
    y = asnumpy(x)
    assert isinstance(y, MultiVector)  # A multivector is a container of tensors, not a tensor itself.
    assert y.keys() == x.keys()
    assert y.shape == x.shape
    assert isinstance(y.values(), np.ndarray)
    assert allclose(y.values(), x.values())

    y_alt = asnumpy(x.map(lambda v: v))
    assert y_alt.shape == x.shape
    assert all(isinstance(v, np.ndarray) for v in y_alt.values())


def test_parse_shape(xp):
    """ parse_shape only sees the array dimensions, not the blades. """
    x = make_mv(xp, (3, 4))
    assert parse_shape(x, 'a b') == {'a': 3, 'b': 4}
    assert parse_shape(x, 'a _') == {'a': 3}


def test_pack_unpack(xp):
    inputs = [xp.zeros((2, 3, 5)), xp.zeros((2, 3, 7, 5)), xp.zeros((2, 3, 7, 9, 5))]
    mvs = [alg.vector(i) for i in inputs]
    assert [mv.shape for mv in mvs] == [(3, 5), (3, 7, 5), (3, 7, 9, 5)]
    packed, ps = pack(mvs, 'i * k')  # i matches 3, k matches 5.
    assert packed.shape == (3, 71, 5)
    assert is_array_api_obj(packed.values())
    inputs_unpacked = unpack(packed, ps, 'i * k')
    assert inputs_unpacked == mvs


def test_pack_unpack_list_values(xp):
    """ pack & unpack also work when the coefficients are a list of arrays. """
    mvs = [alg.vector(xp.zeros((2, 3, 5))).map(lambda v: v),
           alg.vector(xp.zeros((2, 3, 7, 5))).map(lambda v: v)]
    packed, ps = pack(mvs, 'i * k')
    assert packed.shape == (3, 8, 5)
    assert isinstance(packed.values(), list)
    unpacked = unpack(packed, ps, 'i * k')
    assert unpacked == mvs

def test_pack_unpack_inhomogenous(xp):
    """ Pack should work for inhomogenous mvs so long as they are the same type. """
    mvs = [alg.vector(xp.ones((2, 3, 5))).map(lambda v: v),
           alg.vector(e1=xp.ones((3, 7, 5)))]
    packed, ps = pack(mvs, 'i * k')
    assert packed.shape == (3, 8, 5)
    assert isinstance(packed.values(), list)
    unpacked = unpack(packed, ps, 'i * k')
    assert [x.shape for x in unpacked] == [mv.shape for mv in mvs]

    # The keys are relaxed to those of the mv with the most keys, the mv that had all of them
    # comes back unharmed, and the blade the other one lacked is zero.
    assert type(packed) is type(mvs[0]) and packed.keys() == mvs[0].keys()
    assert all(x.keys() == packed.keys() for x in unpacked)
    assert unpacked[0] == mvs[0]
    assert allclose(unpacked[1].e1, mvs[1].e1) and not xp.any(unpacked[1].e2)

    # Those zeros are made from the coefficients of the mv that is missing a blade, so they
    # follow its dtype and are unaffected by any NaNs it happens to carry.
    nan = alg.vector(e1=xp.full((3, 7, 5), xp.nan, dtype=xp.float32))
    packed, ps = pack([mvs[0].map(lambda v: xp.astype(v, xp.float32)), nan], 'i * k')
    assert all(v.dtype == xp.float32 for v in packed.values())
    assert not xp.any(unpack(packed, ps, 'i * k')[1].e2)


def test_pack_unpack_inhomogenous_asarray(xp):
    """ Inhomogenous mvs that all hold their coefficients as one array give a result that does too. """
    sparse = alg.vector(e1=xp.ones((3, 7, 5)))
    sparse = type(sparse).fromkeysvalues(alg, sparse.keys(), xp.stack(list(sparse.values())))
    packed, ps = pack([alg.vector(xp.ones((2, 3, 5))), sparse], 'i * k')
    assert is_array_api_obj(packed.values())
    assert tuple(packed.values().shape) == (2, 3, 8, 5)  # The blade axis survives the zero filling.
    assert not xp.any(unpack(packed, ps, 'i * k')[1].e2)


def test_pack_different_types(xp):
    """ Relaxing the keys is only allowed within one type, since the type fixes their meaning. """
    with pytest.raises(TypeError):
        pack([alg.vector(xp.ones((2, 3, 5))), alg.bivector(xp.ones((1, 3, 5)))], 'i * k')

def test_einsum(xp):
    # einsum is not part of the array API, but every library array-api-compat wraps has it.
    vec_of_matrices = randn(xp, 2, 10, 10)
    vec = alg.vector(vec_of_matrices)
    trace = einsum(vec, "i i ->")
    assert trace.shape == ()
    assert isinstance(trace, MultiVector)
    assert is_array_api_obj(trace.values())
    assert allclose(trace.values(), xp.einsum("zii->z", vec_of_matrices))

    weight = randn(xp, 10, 20)
    matmul = einsum(vec, weight, "i j, j k -> i k")
    assert matmul.shape == (10, 20)
    assert isinstance(matmul, MultiVector)
    assert is_array_api_obj(matmul.values())
    assert allclose(matmul.values(), xp.einsum("zij,jk->zik", vec_of_matrices, weight))

    # Patterns which spell out the batch dimensions with an ellipsis keep working,
    # the ellipsis simply no longer has to swallow the blade axis.
    assert einsum(vec, "... i i -> ...") == trace
    assert einsum(vec, weight, "... j, j k -> ... k") == matmul


def test_einsum_types(xp):
    """ The type of the multivector is preserved, and list-valued multivectors are supported. """
    vec = alg.vector(randn(xp, 2, 10))
    weight = randn(xp, 10, 20)
    matmul = einsum(vec, weight, "i, i j -> j")
    assert type(matmul) is type(vec)
    assert matmul.keys() == vec.keys()
    assert matmul.shape == (20,)

    vec_alt = vec.map(lambda v: v)  # values are a list of arrays instead of one array.
    matmul_alt = einsum(vec_alt, weight, "i, i j -> j")
    assert matmul_alt == matmul


def test_einsum_multiple_multivectors(xp):
    """ Multivectors are combined blade by blade, and scalars multiply every blade. """
    x = alg.vector(randn(xp, 2, 10))
    y = alg.vector(randn(xp, 2, 10))
    prod = einsum(x, y, "i, i ->")
    assert prod.shape == ()
    assert prod.keys() == x.keys()
    assert allclose(prod.values(), xp.einsum("zi,zi->z", x.values(), y.values()))

    s = alg.scalar(randn(xp, 1, 10))
    scaled = einsum(x, s, "i, i -> i")
    assert scaled.keys() == x.keys()
    assert allclose(scaled.values(), x.values() * s.values())

    # Scalars are also fine as the leading operand.
    assert einsum(s, x, "i, i -> i") == scaled

    with pytest.raises(TypeError):
        einsum(x, alg.bivector(randn(xp, 1, 10)), "i, i -> i")

test_pack_unpack_config = [
    # (dimensions of a coefficient beyond the blade axis, pattern, expected shape)
    ((), '*', (3,)),
    ((4,), '* n', (3, 4)),
    ((4,), 'n *', (4, 3)),
]
@pytest.mark.parametrize("extra_shape, pattern, expected_shape", test_pack_unpack_config)
def test_pack_unpack_list(xp, extra_shape, pattern, expected_shape):
    """Test pack and unpack with a list of multivectors and different patterns."""
    mvs = [alg.vector(randn(xp, 2, *extra_shape)) for _ in range(3)]
    packed, ps = pack(mvs, pattern)
    assert packed.shape == expected_shape
    assert isinstance(packed, MultiVector)
    assert is_array_api_obj(packed.values())
    unpacked = unpack(packed, ps, pattern)
    assert isinstance(unpacked, list)
    assert unpacked == mvs

test_stack_config = [
    # (are the coefficients arrays or plain python floats, dimensions beyond the blade axis, expected shape)
    (True, (), (3,)),
    (False, (), (3,)),
    (True, (4,), (3, 4)),
    (False, (4,), (3, 4)),
]
@pytest.mark.parametrize("asarray, extra_shape, expected_shape", test_stack_config)
def test_stack(xp, asarray, extra_shape, expected_shape):
    """Test kingdon.stack with a list of multivectors and different stack functions."""
    def coefficients(n):
        """ The coefficients of one multivector, as an array or as a nested list of floats. """
        return randn(xp, n, *extra_shape) if asarray else np.random.randn(n, *extra_shape).tolist()
    stack_func = xp.stack if asarray else list

    mvs = [alg.vector(coefficients(2)) for _ in range(3)]
    stacked = stack(mvs, stack_func=stack_func)
    assert stacked.shape == expected_shape
    assert isinstance(stacked, MultiVector)
    assert is_array_api_obj(stacked.values()) == asarray
    assert list(stacked) == mvs
    different_mv = alg.evenmv(coefficients(2))
    with pytest.raises(TypeError, match='type'):
        stack([*mvs, different_mv])
    with pytest.raises(TypeError, match='shape'):
        stack([stacked, alg.vector(coefficients(2))])

    # Within one type the keys are relaxed to their union, where the blades an input does
    # not have contribute zeros.
    sparse = alg.vector(e1=coefficients(2)[0])
    relaxed = stack([*mvs, sparse], stack_func=stack_func)
    assert relaxed.keys() == mvs[0].keys()
    assert relaxed.shape == (len(mvs) + 1, *expected_shape[1:])  # The zeros are shaped correctly.
    assert allclose(relaxed[len(mvs)].e1, sparse.e1)
    assert not to_numpy(relaxed[len(mvs)].e2).any()

def test_asarray_algebra(xp):
    # Use an array everywhere
    alg = Algebra(2, values_asarray=partial(values_asarray, xp))
    x = alg.vector(e1=2, e2=3)
    y = alg.vector([4, 5])
    assert is_array_api_obj(x.values())
    assert is_array_api_obj(y.values())
    z = x * y
    assert is_array_api_obj(z.values())

    # Use arrays only for custom operator
    alg = Algebra(2)
    x = alg.vector(e1=2, e2=3)
    y = alg.vector([4, 5])
    assert isinstance(x.values(), list)
    assert isinstance(y.values(), list)
    z = x * y
    assert isinstance(z.values(), list)

    @alg.add_operator(values_asarray=partial(values_asarray, xp))
    def my_func(x, y):
        return x + y
    w = my_func(x, y)
    assert is_array_api_obj(w.values())

    # Inhomogenous data
    x = alg.vector(xp.asarray([[1., 2.], [3., 4.]]))
    s = alg.scalar(xp.asarray([1.0]))
    with pytest.raises(STACK_ERRORS):
        my_func(x, s)

    @alg.add_operator(values_asarray=partial(values_asarray, xp, broadcast=True))
    def my_broadcasted_func(x, y):
        return x + y
    # Inhomogenous data does work when combined with broadcast.
    w = my_broadcasted_func(x, s)
    assert is_array_api_obj(w.values())
    assert w.shape == (2,)


def test_asarray_raw(xp):
    """ values_asarray is applied to numeric values, but symbolic values are left alone. """
    alg = Algebra(2, values_asarray=partial(values_asarray, xp, broadcast=True))

    # Numeric multivectors are cast to arrays, wherever they are created.
    assert is_array_api_obj(alg.vector([1., 2.]).values())
    assert is_array_api_obj((alg.vector([1., 2.]) * alg.vector([3., 4.])).values())
    assert is_array_api_obj(alg.blades.e1.values())

    # Symbolic multivectors keep their values as a list, because casting sympy
    # expressions to an array would give an array of dtype object.
    x = alg.multivector(name='x')
    assert x.issymbolic
    assert isinstance(x.values(), list)

    # Copies are exact replicas, so they keep the values as they are.
    assert copy.copy(x).values() is x.values()
    assert isinstance(copy.deepcopy(x).values(), list)
    assert copy.deepcopy(x) == x

    # Multivectors without any values are created without complaint.
    assert len(alg.vector([1., 2.]).filter(lambda v: False).values()) == 0


def test_136(pga3d):
    """ Multivectors should be compatible with numpy.broadcast. """
    nb_points = 10
    xyz = np.random.rand(3, nb_points)
    point = (pga3d.evector(xyz) + pga3d.blades.e0).dual()
    vector = pga3d.evector(e1=4.0, e2=8.0, e3=-1.2)
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

def test_newaxis_vs_stack(xp):
    pga = Algebra.fromname("3DPGA", values_asarray=partial(values_asarray, xp, broadcast=True))
    N, M = 4, 5
    point_vals = rand(xp, N, 3)
    line_vals = rand(xp, M, 6)
    points = pga.point(einops.rearrange(point_vals, '... blades -> blades ...'))
    bivectors = pga.bivector(einops.rearrange(line_vals, '... blades -> blades ...'))
    lines = bivectors.normalized()
    # Check if the broadcasting results in the same as brute force.
    projected_points = points[:, None] @ lines[None, :]
    assert projected_points == stack([p @ lines for p in points])
    # Check if the values are just a view of the original arrays in case of numpy or torch.
    # In general the array API has no notion of a view.
    if is_numpy_namespace(xp):
        assert points.values().base is point_vals
        assert bivectors.values().base is line_vals
    elif is_torch_namespace(xp):
        assert points.values()._base is point_vals
        assert bivectors.values()._base is line_vals
