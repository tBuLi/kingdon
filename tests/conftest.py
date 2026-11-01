import pytest


@pytest.fixture(params=[None, 'njit'])
def wrapper(request):
    """ Algebra fixtures that take this run once as plain python and once compiled with numba. """
    return request.param and pytest.importorskip('numba').njit
