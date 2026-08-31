import pytest
import itertools

from sympy import symbols, Symbol

from kingdon.operator_dict import OperatorDict, UnaryOperatorDict
import kingdon.operators as ops
from kingdon.polynomial import RationalPolynomial
from kingdon import Algebra, MultiVector, stack


def test_operator_dict():
    alg = Algebra(2)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')

    gp = OperatorDict('gp', codegen=ops.gp, algebra=alg)
    assert gp.codegen_input_types == {'x': MultiVector, 'y': MultiVector}
    assert len(gp) == 0
    with pytest.raises(TypeError):
        gp[x, y] = 2
    xy = gp(x, y)
    assert len(gp) == 1  # size of gp has grown by one
    assert (x, y) in gp

    inv = UnaryOperatorDict('inv', codegen=ops.inv, algebra=alg)
    assert len(inv) == 0
    xinv = inv(x)
    assert len(inv) == 1
    assert x in inv


@pytest.mark.parametrize('codegen_symbolcls', [RationalPolynomial.fromname, Symbol, None], ids=['RationalPolynomial', 'Symbol', 'Numerical'])
def test_codegen_weights(codegen_symbolcls):
    """ In geometric product layers one needs to be able to provide weights as an array of scalars. """
    alg = Algebra(2)

    @alg.jit(symbolic=codegen_symbolcls is not None, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp(x, y, weights: MultiVector[10]):
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9 = weights
        X0, X1, X2 = (x.grade(g) for g in range(alg.d + 1))
        Y0, Y1, Y2 = (y.grade(g) for g in range(alg.d + 1))
        return w0*X0*Y0 + w3*(X1|Y1) + w7*X2*Y2 \
            + w1*X0*Y1 + w4*X1*Y0 + w5*X1*Y2 + w8*X2*Y1 \
            + w2*X0*Y2 + w6*(X1^Y1) + w9*X2*Y0

    assert weighted_gp.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    assert weighted_gp.codegen_output_type == MultiVector
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    ws = symbols('w:10')
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = ws
    weights = alg.scalar(e=ws)
    x0, x1, x2 = x.grade(0), x.grade(1), x.grade(2)
    y0, y1, y2 = y.grade(0), y.grade(1), y.grade(2)
    weighted_gp_output = weighted_gp(x, y, weights)
    assert not weighted_gp_output - (w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0)

    if codegen_symbolcls is None: return  # For the numerical case, the functions below are (not yet) supported.

    @alg.jit(symbolic=codegen_symbolcls is not None, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp_grad_weights(x, y, weights: MultiVector[10]) -> MultiVector[10]:
        """ Output a single mv of shape (coeff, 10). These are all stacked with the same shape, so zeros are not eliminated."""
        weighted_gp_output = weighted_gp(x, y, weights)
        return stack([weighted_gp_output.map(lambda v: v.diff(wi)) for wi in weights.e])

    assert weighted_gp_grad_weights.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    assert weighted_gp_grad_weights.codegen_output_type == (MultiVector, 10)
    grad_weights = weighted_gp_grad_weights(x, y, weights)
    for wi, grad_w in zip(weights.e, grad_weights):
        assert grad_w == weighted_gp_output.map(lambda v: v.diff(wi))

    @alg.jit(symbolic=codegen_symbolcls is not None, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp_grad(x, y, weights: MultiVector[10], go) -> MultiVector[18]:
        syms: list[Symbol] = [*x.values(), *y.values(), *weights.e]
        wgp_output = weighted_gp(x, y, weights)
        go_wgp = go.sp(wgp_output)  # sp -> scalar product
        return stack([go_wgp.map(lambda v: v.diff(s)) for s in syms])

    assert weighted_gp_grad.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10), 'go': MultiVector}
    assert weighted_gp_grad.codegen_output_type == (MultiVector, 18)
    go = alg.multivector(name='go')
    grads = weighted_gp_grad(x, y, weights, go)
    assert grads.keys() == (0,)  # scalar
    assert grads.shape == (18,)
    go_wgp = go.sp(weighted_gp_output)
    for s, grad in zip([*x.values(), *y.values(), *weights.e], grads.e):
        assert not (grad - go_wgp.map(lambda v: v.diff(s)).e).expand()

    # Test non-scalar shaped multivector type-hint
    alg2 = Algebra(2)

    @alg2.jit(symbolic=codegen_symbolcls is not None, codegen_symbolcls=codegen_symbolcls)
    def reduce_gp(mvs: MultiVector[2]):
        mv1, mv2 = mvs
        return mv1*mv2

    assert reduce_gp.codegen_input_types == {'mvs': (MultiVector, 2)}
    assert reduce_gp.codegen_output_type == MultiVector
    x2 = alg2.multivector(name='x')
    y2 = alg2.multivector(name='y')
    mv_symbols = list(zip(x2.values(), y2.values()))
    mvs = alg2.multivector(values=mv_symbols)
    assert reduce_gp(mvs) == x2*y2


@pytest.mark.xfail(reason='Compiling a function that returns a list of multivectors is not supported yet.')
@pytest.mark.parametrize('codegen_symbolcls', [RationalPolynomial.fromname, Symbol], ids=['RationalPolynomial', 'Symbol'])
def test_wgp_list(codegen_symbolcls):
    """Generate a function that returns a list of multivectors."""
    alg = Algebra(2)

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp(x, y, weights: MultiVector[10]):
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = weights
        X0, X1, X2 = (x.grade(g) for g in range(alg.d + 1))
        Y0, Y1, Y2 = (y.grade(g) for g in range(alg.d + 1))
        return w0 * X0 * Y0 + w3 * (X1 | Y1) + w7 * X2 * Y2 \
            + w1 * X0 * Y1 + w4 * X1 * Y0 + w5 * X1 * Y2 + w8 * X2 * Y1 \
            + w2 * X0 * Y2 + w6 * (X1 ^ Y1) + w9 * X2 * Y0

    assert weighted_gp.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    assert weighted_gp.codegen_output_type == MultiVector
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    ws = symbols('w:10')
    weights = alg.scalar(e=ws)
    weighted_gp_output = weighted_gp(x, y, weights)

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp_grad_weights_list(x, y, weights: MultiVector[10]) -> list[MultiVector]:
        """
        Generate a list of output mv's of different shape. Same content as weighted_gp_grad_weight,
        but zeros can be eliminated here.
        """
        weighted_gp_output = weighted_gp(x, y, weights)
        return [weighted_gp_output.map(lambda v: v.diff(wi)).filter() for wi in weights.e]

    assert weighted_gp_grad_weights_list.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    assert weighted_gp_grad_weights_list.codegen_output_type == list[MultiVector]
    grad_weights_list = weighted_gp_grad_weights_list(x, y, weights)
    assert isinstance(grad_weights_list, list)
    for wi, grad_w in zip(weights.e, grad_weights_list):
        assert grad_w == weighted_gp_output.map(lambda v: v.diff(wi))


@pytest.mark.parametrize('codegen_symbolcls', [RationalPolynomial.fromname, Symbol], ids=['RationalPolynomial', 'Symbol'])
def test_codegen_wgp_generic(codegen_symbolcls):
    """
    Similar to test_codegen_weights, but with an unspecified number of weights.
    This is meant to test the type-hinting syntax for a new dimension of unknown size.
    """
    alg = Algebra(2)

    def number_of_weights_wgp(X: MultiVector, Y: MultiVector) -> int:
        i = 0
        for gx, gy in itertools.product(X.grades, Y.grades):
            Z = X.grade(gx) * Y.grade(gy)
            i += len(Z.grades)
        return i

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls)
    def wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
        """
        Compute the weighted geometric product between X and Y.
        The multivectors are mutiplied grade-wise, and a unique weight
        is applied to each grade in the output.
        """
        tot = 0
        i = 0
        for gx, gy in itertools.product(X.grades, Y.grades):
            Z = X.grade(gx) * Y.grade(gy)
            for gz in Z.grades:
                tot += weights[i] * Z.grade(gz)
                i += 1
        return tot


    assert wgp.codegen_input_types == {'X': MultiVector, 'Y': MultiVector, 'weights': (MultiVector, None)}
    assert wgp.codegen_output_type == MultiVector
    # assert number_of_weights_wgp.codegen_input_types == {'X': MultiVector, 'Y': MultiVector}
    # assert number_of_weights_wgp.codegen_output_type == int

    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    num_weights = number_of_weights_wgp(x, y)
    ws = symbols(f'w:{num_weights}')
    w0, w1, w2, w4, w3, w6, w5, w9, w8, w7 = ws
    weights = alg.scalar(e=ws)
    x0, x1, x2 = x.grade(0), x.grade(1), x.grade(2)
    y0, y1, y2 = y.grade(0), y.grade(1), y.grade(2)
    compiled_expr = wgp[x, y, weights]
    wgp_output = wgp(x, y, weights)
    assert wgp_output == w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0


@pytest.mark.parametrize('codegen_symbolcls', [RationalPolynomial.fromname, Symbol], ids=['RationalPolynomial', 'Symbol'])
def test_codegen_set(codegen_symbolcls):
    alg = Algebra(2)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    z = alg.multivector(name='z')
    ws = symbols('w:10')
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = ws
    weights = alg.scalar(e=ws)

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls)
    def set_gp(x, y, z):
        _z = x*y
        z.set(_z)

    res = set_gp(x, y, z)
    assert res == None
    assert z == x*y

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls)
    def weighted_gp_set(x, y, weights: MultiVector[10], z):
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9 = weights
        X0, X1, X2 = (x.grade(g) for g in range(alg.d + 1))
        Y0, Y1, Y2 = (y.grade(g) for g in range(alg.d + 1))
        z.set(w0*X0*Y0 + w3*(X1|Y1) + w7*X2*Y2 \
            + w1*X0*Y1 + w4*X1*Y0 + w5*X1*Y2 + w8*X2*Y1 \
            + w2*X0*Y2 + w6*(X1^Y1) + w9*X2*Y0)

    x0, x1, x2 = x.grade(0), x.grade(1), x.grade(2)
    y0, y1, y2 = y.grade(0), y.grade(1), y.grade(2)
    res = weighted_gp_set(x, y, weights, z)
    assert res == None
    assert z == w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0


@pytest.mark.parametrize('codegen_symbolcls', [RationalPolynomial.fromname, Symbol], ids=['RationalPolynomial', 'Symbol'])
def test_codegen_printer(codegen_symbolcls):
    alg = Algebra(2)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')

    from sympy.printing.lambdarepr import LambdaPrinter
    from kingdon.codegen import KingdonPrinter

    class MyPrinter(LambdaPrinter):
        pass

    class MyEvaluatorPrinter(KingdonPrinter):
        pass

    def my_wrapper(func):
        return func

    my_printer = MyPrinter()
    my_func_printer = MyEvaluatorPrinter(my_printer)

    @alg.jit(symbolic=True, codegen_symbolcls=codegen_symbolcls, printer=my_printer, func_printer=my_func_printer, wrapper=my_wrapper)
    def my_gp(x, y):
        return x*y
    res = my_gp(x, y)
    assert res == x*y
    assert my_gp.printer == my_printer
    assert my_gp.func_printer == my_func_printer
    assert my_gp.wrapper == my_wrapper
