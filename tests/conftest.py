import pytest

from kingdon import Algebra


@pytest.fixture(params=[None, 'njit'])
def wrapper(request):
    """
    Algebra fixtures that take this run once as plain python and once compiled with numba.
    Tests that cannot run under numba opt out with :code:`@pytest.mark.parametrize('wrapper', [None])`.
    """
    return request.param and pytest.importorskip('numba').njit


@pytest.fixture
def ga101(wrapper):
    return Algebra(signature=[1, 0], start_index=1, wrapper=wrapper)


@pytest.fixture
def ga201(wrapper):
    return Algebra(signature=[1, 1, 0], start_index=1, wrapper=wrapper)


@pytest.fixture
def ga301(wrapper):
    return Algebra(signature=[1, 1, 1, 0], start_index=1, wrapper=wrapper)


@pytest.fixture
def alg201(wrapper):
    return Algebra(2, 0, 1, wrapper=wrapper)


@pytest.fixture
def alg301(wrapper):
    return Algebra(3, 0, 1, wrapper=wrapper)


@pytest.fixture
def alg211(wrapper):
    return Algebra(2, 1, 1, wrapper=wrapper)


@pytest.fixture
def pga2d(wrapper):
    return Algebra.fromname('2DPGA', wrapper=wrapper)


@pytest.fixture
def pga3d(wrapper):
    return Algebra.fromname('3DPGA', wrapper=wrapper)


@pytest.fixture
def vga2d(wrapper):
    return Algebra(2, wrapper=wrapper)


@pytest.fixture
def vga3d(wrapper):
    return Algebra(3, wrapper=wrapper)


@pytest.fixture
def vga11(wrapper):
    return Algebra(1, 1, wrapper=wrapper)


@pytest.fixture
def R4(wrapper):
    return Algebra(4, wrapper=wrapper)


@pytest.fixture
def R6(wrapper):
    return Algebra(6, wrapper=wrapper)


@pytest.fixture
def sta(wrapper):
    return Algebra(3, 1, wrapper=wrapper)
