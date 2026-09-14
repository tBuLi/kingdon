"""
Tests for the torch backend, see :doc:`docs/backends/torch.rst`.

Every test here stands for a claim that page makes, and there is deliberately nothing beyond that:
what is not promised is free to change.
"""
import pytest

torch = pytest.importorskip('torch')
einops = pytest.importorskip('einops')

from torch import nn

from kingdon import Algebra
from kingdon.torch_backend import values_asarray


@pytest.fixture
def alg():
    return Algebra(3, backend='torch')


@pytest.fixture
def x(alg):
    """ A vector of shape (4, 5). None of 3 blades, 4 and 5 coincide, so a swapped axis shows. """
    return alg.vector(torch.randn(3, 4, 5))


def same(a, b):
    """ Whether two multivectors hold the same coefficients, in either storage form. """
    return a.keys() == b.keys() and all(torch.allclose(u, v) for u, v in zip(a.values(), b.values()))


# The docs name Linear, the activation functions, LayerNorm and Dropout.
MODULES = [nn.Linear(5, 7), nn.GELU(), nn.LayerNorm(5), nn.Dropout(0.1),
           nn.Sequential(nn.Linear(5, 8), nn.GELU())]


@pytest.mark.parametrize('module', MODULES, ids=[type(m).__name__ for m in MODULES])
def test_a_torch_nn_module_acts_on_all_blades_at_once(x, module):
    """
    "A torch function is handed mv.values(), and its result becomes the coefficients of the
    multivector that comes back", and the blade axis leads, so a module never reaches it.
    """
    module = module.eval()                                 # so that Dropout is deterministic
    out = module(x)
    assert type(out) is type(x) and out.keys() == x.keys()
    assert torch.equal(out.values(), module(x.values()))   # the very same call on the coefficients
    assert out.shape == module(x.values()).shape[1:]


def test_shape_hides_the_blade_axis(x):
    """ A multivector over an array of shape (blades, ..., channels) has shape (..., channels). """
    assert x.values().shape == (3, 4, 5)
    assert x.shape == (4, 5) and x.ndim == 2


def test_a_dim_counts_the_axes_of_the_coefficients(x):
    """ Summing the channels is allowed; summing the blades is not. """
    assert torch.sum(x, -1).shape == (4,)
    assert same(torch.sum(x, -1), x.map(lambda v: torch.sum(v, -1)))
    with pytest.raises(TypeError, match='addressed the blade axis'):
        torch.sum(x, 0)


def test_einops_addresses_the_axes_of_the_multivector(x):
    """ einops patterns refer to mv.shape. Note that nothing here imports the einops backend. """
    assert einops.reduce(x, 'a b -> b', 'sum').shape == (5,)
    assert einops.rearrange(x, 'a b -> b a').shape == (5, 4)
    assert einops.pack([x, x], '* b')[0].shape == (8, 5)
    assert einops.einsum(x, torch.randn(5, 7), 'a b, b c -> a c').shape == (4, 7)


#: The operator table of the docs: symbol -> algebra operator, and torch's name where it has one.
OPERATORS = {'+': ('add', 'add'), '-': ('sub', 'sub'), '*': ('gp', 'mul'), '/': ('div', 'div'),
             '@': ('proj', 'matmul'), '|': ('ip', None), '^': ('op', None), '&': ('rp', None),
             '>>': ('sw', None)}


@pytest.mark.parametrize('symbol, operator, spelling',
                         [(s, *v) for s, v in OPERATORS.items()], ids=list(OPERATORS))
def test_every_spelling_of_an_operator_is_the_algebra_operator(x, symbol, operator, spelling):
    """ However torch spells an operator it is handed to the algebra, on either side of it. """
    t = torch.randn(5)
    reference = getattr(x.algebra, operator)
    assert same(eval(f'x {symbol} t'), reference(x, t))
    assert same(eval(f't {symbol} x'), reference(t, x))     # torch wins this side of it
    if spelling:                                            # as a torch function, and as a method
        assert same(getattr(torch, spelling)(x, t), reference(x, t))
        assert same(getattr(x, spelling)(t), reference(x, t))


def test_the_operators_where_coefficient_by_coefficient_would_not_do(x):
    """ The unary row of the table, and the reason the operators are an exception at all. """
    assert same(torch.neg(x), -x)
    y = x.algebra.bivector(torch.randn(3, 4, 5))
    assert torch.add(x, y).keys() == (x + y).keys() != x.keys()    # the union of the keys
    assert same(torch.multiply(x, y), x * y)                       # the aliases go along


def test_no_other_torch_name_means_geometric_algebra(alg, x):
    """ torch.exp exponentiates the coefficients; mv.exp() is the exponential of the multivector. """
    assert same(torch.exp(x), x.map(torch.exp))
    assert 0 in alg.bivector(e12=torch.tensor(0.3)).exp().keys()   # a rotor has a scalar part


def test_gradients_flow_through_a_module(alg):
    """ The layer from the docs: a learned rotor, then a Linear. """
    class GALayer(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.lin = nn.Linear(n_in, n_out)
            self.bivector = nn.Parameter(torch.randn(3) * 0.1)

        def forward(self, p):
            b = self.bivector
            R = alg.bivector(e12=b[0], e13=b[1], e23=b[2]).exp()
            return torch.relu(self.lin(R >> p))

    m = GALayer(5, 7)
    out = m(alg.vector(torch.randn(3, 4, 5)))
    assert out.shape == (4, 7)
    out.values().sum().backward()                           # mv.values() is one tensor
    assert all(p.grad is not None and (p.grad != 0).any() for p in m.parameters())


def test_tensor_methods_and_the_names_that_come_first(x):
    """ Methods go the same way as the free functions, and kingdon's own names still win. """
    assert x.detach().to(torch.float64).dtype is torch.float64
    assert same(x.relu(), torch.relu(x))
    assert torch.equal(x.e1, x.values()[0]) and x.ndim == 2
    with pytest.raises(AttributeError, match='no attribute or basis blade foobar'):
        x.foobar
    with pytest.raises(AttributeError, match='no attribute or basis blade relu'):
        Algebra(3).vector([1.0, 2.0, 3.0]).relu             # no tensors: not torch's business


def test_the_backend_sets_values_asarray(x):
    """ It keeps the coefficients in a single tensor; your own values_asarray is left alone. """
    assert x.algebra.values_asarray is values_asarray
    assert isinstance((x * x).values(), torch.Tensor)       # the operators got it too
    mine = list.copy                                        # anything that survives a plain [1]
    assert Algebra(2, backend='torch', values_asarray=mine).values_asarray is mine
    with pytest.raises(ValueError, match='Unknown backend'):
        Algebra(2, backend='torhc')


def test_the_backend_works_over_whatever_values_asarray_leaves():
    """ Since your own values_asarray is left alone, coefficients may still be a plain list. """
    v = Algebra(3).vector([torch.randn(4, 5) for _ in range(3)])
    assert isinstance(v.values(), list)
    assert nn.Linear(5, 7)(v).shape == (4, 7)


def test_torch_compile(x):
    """
    The shipped values_asarray was designed to be compatible with torch.compile, and
    wrapper=torch.compile applies it to every function kingdon generates. One test for both, since
    the first compile in a process pays for torch's compiler either way.
    """
    assert same(torch.compile(lambda a: a * a)(x), x * x)
    compiled = Algebra(3, backend='torch', wrapper=torch.compile)
    assert same(compiled.vector(x.values()) * compiled.vector(x.values()), x * x)


def test_a_kingdon_without_the_backend_never_meets_torch():
    """
    "backend='torch' is what imports kingdon.torch_backend, and importing that is what puts
    __torch_function__ on a multivector". In a subprocess, since this module has asked already.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        import torch
        from kingdon import Algebra
        from kingdon.multivector import MultiVector

        alg = Algebra(3)                                    # no backend asked for
        v = alg.vector([torch.randn(4) for _ in range(3)])
        assert not hasattr(MultiVector, '__torch_function__')
        assert 'kingdon.torch_backend' not in sys.modules and 'einops' not in sys.modules
        assert type(torch.randn(4) * v).__name__ == 'Vector'   # via MultiVector.__rmul__

        Algebra(3, backend='torch')                         # and now it should
        assert hasattr(MultiVector, '__torch_function__')
        assert alg.vector(torch.randn(3, 4)).relu().shape == (4,)
    """)
    assert subprocess.run([sys.executable, '-c', script], capture_output=True).returncode == 0


def test_symbolic_codegen_is_unaffected(alg):
    """
    Not a claim of the torch page, but the one interaction with the rest of kingdon worth pinning:
    a symbolic multivector never reaches values_asarray, so codegen still works over the backend.
    """
    assert (alg.vector(name='x') * alg.vector(name='y')).issymbolic

    def double(x):
        return x + x

    alg.add_operator(double, symbolic=True)
    x = alg.vector(torch.randn(3, 4, 5))
    assert same(alg.registry['double'](x), x + x)
