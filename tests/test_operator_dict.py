import pytest

from sympy import symbols

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
    def weighted_ip(x, y, weights: MultiVector[10]):
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9 = weights
        X0, X1, X2 = (x.grade(g) for g in range(alg.d + 1))
        Y0, Y1, Y2 = (y.grade(g) for g in range(alg.d + 1))
        return w0*X0*Y0 + w3*(X1|Y1) + w7*X2*Y2 \
            + w1*X0*Y1 + w4*X1*Y0 + w5*X1*Y2 + w8*X2*Y1 \
            + w2*X0*Y2 + w6*(X1^Y1) + w9*X2*Y0

    assert weighted_ip.codegen_types == {'x': MultiVector, 'y': MultiVector, 'weights': (MultiVector, 10)}
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    ws = symbols('w0, w1, w2, w3, w4, w5, w6, w7, w8, w9')
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = ws
    weights = alg.scalar(e=ws)
    x0, x1, x2 = x.grade(0), x.grade(1), x.grade(2)
    y0, y1, y2 = y.grade(0), y.grade(1), y.grade(2)
    keys_out, func = weighted_ip[x, y, weights]
    res = weighted_ip(x, y, weights)
    assert res == w0*x0*y0 + w3*(x1|y1) + w7*x2*y2 + w1*x0*y1 + w4*x1*y0 + w5*x1*y2 + w8*x2*y1 + w2*x0*y2 + w6*(x1^y1) + w9*x2*y0