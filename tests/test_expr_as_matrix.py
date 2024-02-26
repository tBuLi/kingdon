from kingdon import Algebra
from kingdon.matrixreps import expr_as_matrix
from sympy import Matrix, pprint


def test_expr_as_matrix():
    alg = Algebra(3, 0)
    x = alg.vector(name='x')
    B = alg.bivector(name='B')

    # Test for the matrix rep of the commutator. (Grade preserving)
    A, y = expr_as_matrix(alg.cp, B, x)
    B12, B13, B23 = B.values()
    assert A == Matrix([[0, B12, B13], [-B12, 0, B23], [-B13, -B23, 0]])
    assert y == B.cp(x)
    assert [alg.bin2canon[k] for k in y.keys()] == ['e1', 'e2', 'e3']

    # Test for the matrix rep of the commutator. (Grade preserving, only some output)
    X = alg.vector(e1=1, e3=1)
    A, y = expr_as_matrix(alg.cp, B, x, res_like=X)
    B12, B13, B23 = B.values()
    assert A == Matrix([[0, B12, B13], [-B13, -B23, 0]])
    assert [alg.bin2canon[k] for k in y.keys()] == ['e1', 'e3']

    # Test for the matrix rep of the anti-commutator.
    A, y = expr_as_matrix(alg.acp, B, x)
    assert A == Matrix([[B23, -B13, B12]])
    assert [alg.bin2canon[k] for k in y.keys()] == ['e123']

    # Test for the matrix rep of conjugation of the e3 plane.
    x = alg.vector(name='x', keys=('e3',))
    A, y = expr_as_matrix(alg.sw, B, x)
    assert A == Matrix([[2*B12*B23], [-2*B12*B13], [B12**2 - B13**2 - B23**2], [0]])
    assert [alg.bin2canon[k] for k in y.keys()] == ['e1', 'e2', 'e3', 'e123']
