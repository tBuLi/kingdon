import itertools

import pytest

from kingdon import Algebra, MultiVector

torch = pytest.importorskip('torch')
pytest.importorskip('triton')
if not torch.cuda.is_available():
    pytest.skip('triton kernels need a gpu', allow_module_level=True)

from kingdon.triton_codegen import triton_lambdify  # noqa: E402


def wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
    tot, i = 0, 0
    for gx, gy in itertools.product(X.grades, Y.grades):
        Z = X.grade(gx) * Y.grade(gy)
        for gz in Z.grades:
            tot += weights[i] * Z.grade(gz)
            i += 1
    return tot


def n_weights(algebra):
    x, y = algebra.multivector(name='x'), algebra.multivector(name='y')
    return sum(len((x.grade(a) * y.grade(b)).grades) for a, b in itertools.product(x.grades, y.grades))


def run(build, dim, tensors, lambdifier, loss=lambda values: (values * values).sum()):
    """
    Evaluate `build` on a fresh torch algebra of dimension `dim`, and return the values of the
    multivector it makes together with the gradients of `loss` with respect to `tensors`.

    `build` is called as :code:`build(algebra, *tensors)` on clones that require grad, so the
    `lambdifier` is the only thing that differs between an eager and a triton run of the same test.
    """
    algebra = Algebra(dim, backend='torch', lambdifier=lambdifier)
    ts = [t.clone().requires_grad_(True) for t in tensors]
    values = build(algebra, *ts).values()
    loss(values).backward()
    return values.detach(), [t.grad for t in ts]


def weighted_gp(algebra, x, y, w):
    algebra.add_operator(wgp, symbolic=True)
    return algebra.registry['wgp'](algebra.multivector(x), algebra.multivector(y), algebra.scalar(e=w))


LAYOUTS = {
    'feature': lambda n, k: [(n, 48, 16), (n, 48, 16), (k, 16)],
    # A fully connected layer: the inputs are shared over the output features and the weights over the batch, so no operand varies along every axis.
    'fully connected': lambda n, k: [(n, 48, 1, 16), (n, 48, 1, 16), (k, 8, 16)],
}


# Batch 48 spans several backward blocks, so a weight gradient that is only right for one block fails here.
@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('dim', [2, 3, 4, 5])
def test_matches_eager(dim, layout):
    algebra = Algebra(dim)
    n, k = len(algebra), n_weights(algebra)
    torch.manual_seed(0)
    tensors = [torch.randn(*shape, device='cuda') for shape in LAYOUTS[layout](n, k)]

    want_values, want_grads = run(weighted_gp, dim, tensors, None)
    got_values, got_grads = run(weighted_gp, dim, tensors, triton_lambdify)

    torch.testing.assert_close(got_values, want_values, rtol=1e-5, atol=1e-4)
    for name, want, got in zip('xyw', want_grads, got_grads):
        scale = max(want.abs().max().item(), 1.0)
        assert (got - want).abs().max().item() / scale < 1e-5, f'gradient of {name}'


@pytest.mark.parametrize('dim', [2, 3])
def test_divide_by_a_multivector(dim):
    """A quotient is still rational, so it gets a kernel rather than the fallback."""
    algebra = Algebra(dim, backend='triton')
    n = len(algebra)
    torch.manual_seed(0)
    tensors = [torch.randn(n, 48, 16, device='cuda'), torch.randn(1, 48, 16, device='cuda').abs() + 1.0]

    def divide(alg, x, y):
        return alg.multivector(x) / alg.scalar(e=y[0])

    want, want_grads = run(divide, dim, tensors, None)
    got, got_grads = run(divide, dim, tensors, triton_lambdify)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)
    for want_grad, got_grad in zip(want_grads, got_grads):
        torch.testing.assert_close(got_grad, want_grad, rtol=1e-4, atol=1e-3)


def test_sqrt():
    """A root is an opaque symbol over a remembered base, so it differentiates and emits."""
    torch.manual_seed(0)
    raw = torch.rand(1, 48, 16, device='cuda') + 1.0

    def root(alg, t):
        return alg.sqrt(alg.scalar(e=t[0]))

    cube = lambda values: (values * values * values).sum()
    want, (want_grad,) = run(root, 2, [raw], None, loss=cube)
    got, (got_grad,) = run(root, 2, [raw], triton_lambdify, loss=cube)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(got_grad, want_grad, rtol=1e-4, atol=1e-3)
