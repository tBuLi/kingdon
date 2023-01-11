import pytest

from kingdon.operator_dict import OperatorDict, UnaryOperatorDict
from kingdon.codegen import codegen_gp, codegen_inv
from kingdon import Algebra


def test_operator_dict():
    alg = Algebra(2)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')

    gp = OperatorDict('gp', codegen=codegen_gp, algebra=alg)
    assert len(gp) == 0
    with pytest.raises(TypeError):
        gp[(x.keys(), y.keys())] = 2
    xy = gp(x, y)
    assert len(gp) == 1  # size of gp has grown by one
    assert (x.keys(), y.keys()) in gp

    inv = UnaryOperatorDict('inv', codegen=codegen_inv, algebra=alg)
    assert len(inv) == 0
    xinv = inv(x)
    assert len(inv) == 1
    assert x.keys() in inv
