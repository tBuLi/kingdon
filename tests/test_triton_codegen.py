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


def test_compiles_without_graph_breaks():
    """
    torch.compile traces the kernels of a warm operator, forward and backward, in one graph, and again once a second batch size makes the batch symbolic.
    Through inductor, which launches the kernels its own way and, given several tiles, times them with the gradients reset between runs.
    """
    algebra = Algebra(3, backend='triton')
    algebra.add_operator(wgp, symbolic=True)
    n, k = len(algebra), n_weights(algebra)

    def loss(x, y, w):
        values = algebra.registry['wgp'](algebra.multivector(x), algebra.multivector(y), algebra.scalar(e=w)).values()
        return (values * values).sum()

    def grads(fn, tensors):
        ts = [t.clone().requires_grad_(True) for t in tensors]
        fn(*ts).backward()
        return [t.grad for t in ts]

    compiled = torch.compile(loss, fullgraph=True)
    torch.manual_seed(0)
    for batch in (48, 40):
        tensors = [torch.randn(*shape, device='cuda') for shape in LAYOUTS['fully connected'](n, k)]
        tensors[:2] = [t[:, :batch] for t in tensors[:2]]
        for want, got in zip(grads(loss, tensors), grads(compiled, tensors)):
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)


def test_sympy_functions():
    """An operator over sympy symbols may call the functions triton has: flash-clifford's GELU gates before a weighted product, one kernel forward and one backward."""
    import math
    import sympy

    def gated(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
        gate = lambda mv: mv * (0.5 * (1 + sympy.erf(mv.e / math.sqrt(2))))
        return wgp(gate(X), gate(Y), weights)

    torch.manual_seed(0)
    n, k = len(Algebra(2)), n_weights(Algebra(2))
    tensors = [torch.randn(*shape, device='cuda') for shape in LAYOUTS['feature'](n, k)]
    results = []
    for backend in ('torch', 'triton'):
        alg = Algebra(2, backend=backend)
        alg.add_operator(gated, symbolic=True, codegen_symbolcls=sympy.Symbol)
        ts = [t.clone().requires_grad_(True) for t in tensors]
        x, y, w = alg.multivector(ts[0]), alg.multivector(ts[1]), alg.scalar(e=ts[2])
        values = alg.registry['gated'](x, y, w).values()
        (values * values).sum().backward()
        results.append((values.detach(), [t.grad for t in ts]))
    dispatch = alg.registry['gated'][x, y, w].func
    built = dict(zip(dispatch.__code__.co_freevars, (c.cell_contents for c in dispatch.__closure__)))['built']
    assert built and all(built.values()), 'fell back to torch'
    (want, want_grads), (got, got_grads) = results
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


@pytest.mark.parametrize('features', [(4, 6), (16, 32)])
def test_einops(features):
    """An operator with einops calls is one kernel over blocks of rows: a linear map per grade, a gate and a mean over the features, in registers. Features as many as the card's tl.dot takes contract by one."""
    import sympy
    from einops import einsum, reduce
    from kingdon import Scalar

    def layer(X: MultiVector, W: Scalar[None], b: Scalar) -> MultiVector:
        Y = einsum(X, W[X.gradeidx_of_blades], "... i, o i -> ... o") + b
        Y = Y * sympy.erf(Y.e)
        return Y / (reduce((Y * ~Y).grade(0), "... o -> ... 1", "mean") + 1)

    torch.manual_seed(0)
    i, o = features
    tensors = [torch.randn(4, 48, i, device='cuda'), torch.randn(1, 3, o, i, device='cuda'), torch.randn(1, o, device='cuda')]
    results = []
    for backend in ('torch', 'triton'):
        alg = Algebra(2, backend=backend)
        alg.add_operator(layer, symbolic=True, codegen_symbolcls=sympy.Symbol)
        ts = [t.clone().requires_grad_(True) for t in tensors]
        args = alg.multivector(ts[0]), alg.scalar(e=ts[1][0]), alg.scalar(e=ts[2][0])
        values = alg.registry['layer'](*args).values()
        (values * values).sum().backward()
        results.append((values.detach(), [t.grad for t in ts]))
    dispatch = alg.registry['layer'][args].func
    built = dict(zip(dispatch.__code__.co_freevars, (c.cell_contents for c in dispatch.__closure__)))['built']
    assert built and all(built.values()), 'fell back to torch'
    (want, want_grads), (got, got_grads) = results
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)
    for want_grad, got_grad in zip(want_grads, got_grads):
        torch.testing.assert_close(got_grad, want_grad, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('unrolled', [256, -1])
def test_gather(monkeypatch, unrolled):
    """An einsum over gathered blades and signed weights: spelled out per blade up to the limit of the printer, and looped over by tables beyond, here forced."""
    import einops
    import sympy
    from einops import einsum
    from kingdon import Scalar
    import kingdon.triton_codegen

    monkeypatch.setattr(kingdon.triton_codegen, '_UNROLLED', unrolled)

    def gathered(X: MultiVector, Y: MultiVector, w: Scalar) -> MultiVector:
        J = X.fromkeysvalues(X.algebra, X.keys(), [[(c + 3 * a) % 8 for a in range(8)] for c in range(8)], raw=True)
        P = X.fromkeysvalues(X.algebra, X.keys(), [[(5 * c + 7 * a) % 30 for a in range(8)] for c in range(8)], raw=True)
        return einsum(X.blades, Y.blades[J], einops.pack([w, -w, 0 * w], "* f")[0][P], "a ... f, a ... f, a f -> ... f")

    torch.manual_seed(0)
    tensors = [torch.randn(8, 48, 16, device='cuda'), torch.randn(8, 48, 16, device='cuda'), torch.randn(1, 10, 16, device='cuda')]
    results = []
    for backend in ('torch', 'triton'):
        alg = Algebra(3, backend=backend)
        alg.add_operator(gathered, symbolic=True, codegen_symbolcls=sympy.Symbol)
        ts = [t.clone().requires_grad_(True) for t in tensors]
        args = alg.multivector(ts[0]), alg.multivector(ts[1]), alg.scalar(e=ts[2][0])
        values = alg.registry['gathered'](*args).values()
        (values * values).sum().backward()
        results.append((values.detach(), [t.grad for t in ts]))
    dispatch = alg.registry['gathered'][args].func
    built = dict(zip(dispatch.__code__.co_freevars, (c.cell_contents for c in dispatch.__closure__)))['built']
    assert built and all(built.values()), 'fell back to torch'
    (want, want_grads), (got, got_grads) = results
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)
    for want_grad, got_grad in zip(want_grads, got_grads):
        torch.testing.assert_close(got_grad, want_grad, rtol=1e-4, atol=1e-3)
