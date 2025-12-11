import pytest
import itertools

from sympy import symbols, Symbol

from kingdon.operator_dict import OperatorDict, UnaryOperatorDict
from kingdon.codegen import codegen_gp, codegen_inv
from kingdon import Algebra, MultiVector


def test_operator_dict():
    alg = Algebra(2)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')

    gp = OperatorDict('gp', codegen=codegen_gp, algebra=alg)
    # assert gp.codegen_input_types == {'x': MultiVector, 'y': MultiVector}
    assert len(gp) == 0
    with pytest.raises(TypeError):
        gp[(x, y)] = 2
    xy = gp(x, y)
    assert len(gp) == 1  # size of gp has grown by one
    assert (x, y) in gp

    inv = UnaryOperatorDict('inv', codegen=codegen_inv, algebra=alg)
    assert len(inv) == 0
    xinv = inv(x)
    assert len(inv) == 1
    assert x in inv

def test_codegen_weights():
    """ In geometric product layers one needs to be able to provide weights as an array of scalars. """
    alg = Algebra(2)

    @alg.compile(symbolic=True)
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
    keys_out, func = weighted_gp[x, y, weights]
    weighted_gp_output = weighted_gp(x, y, weights)
    assert weighted_gp_output == w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0

    @alg.compile(symbolic=True, codegen_symbolcls=Symbol)
    def weighted_gp_grad_weights(x, y, weights: MultiVector[10]) -> MultiVector[10]:
        weighted_gp_output = weighted_gp(x, y, weights)
        return [weighted_gp_output.map(lambda v: v.diff(wi)) for wi in weights.e]

    assert weighted_gp_grad_weights.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    assert weighted_gp_grad_weights.codegen_output_type == (MultiVector, 10)
    grad_weights = weighted_gp_grad_weights(x, y, weights)
    for wi, grad_w in zip(weights.e, grad_weights):
        assert grad_w == weighted_gp_output.map(lambda v: v.diff(wi))

    @alg.compile(symbolic=True, codegen_symbolcls=Symbol)
    def weighted_gp_grad(x, y, weights: MultiVector[10], go) -> MultiVector[18]:
        syms: list[Symbol] = [*x.values(), *y.values(), *weights.e]
        wgp_output = weighted_gp(x, y, weights)
        go_wgp = go.sp(wgp_output)  # sp -> scalar product
        return [go_wgp.map(lambda v: v.diff(s)) for s in syms]

    assert weighted_gp_grad.codegen_input_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10), 'go': MultiVector}
    assert weighted_gp_grad.codegen_output_type == (MultiVector, 18)
    go = alg.multivector(name='go')
    grads = weighted_gp_grad(x, y, weights, go)
    assert grads.keys() == (0,)  # scalar
    assert grads.shape == (1, 18)
    go_wgp = go.sp(weighted_gp_output)
    for s, grad in zip([*x.values(), *y.values(), *weights.e], grads.e):
        assert grad == go_wgp.map(lambda v: v.diff(s)).e

def test_codegen_wgp_generic():
    """
    Similar to test_codegen_weights, but with an unspecified number of weights.
    This is meant to test the type-hinting syntax for a new dimension of unknown size.
    """
    alg = Algebra(2)

    @alg.compile(symbolic=True)
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

    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    ws = symbols('w:10')
    w0, w1, w2, w4, w3, w6, w5, w9, w8, w7 = ws
    weights = alg.scalar(e=ws)
    x0, x1, x2 = x.grade(0), x.grade(1), x.grade(2)
    y0, y1, y2 = y.grade(0), y.grade(1), y.grade(2)
    keys_out, func = wgp[x, y, weights]
    wgp_output = wgp(x, y, weights)
    assert wgp_output == w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0