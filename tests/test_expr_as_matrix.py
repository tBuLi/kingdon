from kingdon import Algebra
from kingdon.matrixreps import expr_as_matrix
from sympy import Matrix, pprint


def test_expr_as_matrix():
    alg = Algebra(3, 0)
    x = alg.vector(name='x')
    B = alg.bivector(name='B')

    # Test for the matrix rep of the commutator. (Grade preserving)
    A, rows, columns = expr_as_matrix(alg.cp, B, x)
    B12, B13, B23 = B.values()
    assert A == Matrix([[0, B12, B13], [-B12, 0, B23], [-B13, -B23, 0]])
    assert rows == columns
    assert rows == ['e1', 'e2', 'e3']

    # Test for the matrix rep of the commutator. (Grade preserving, only some output)
    y = alg.vector(e1=1, e3=1)
    A, rows, columns = expr_as_matrix(alg.cp, B, x, res_like=y)
    B12, B13, B23 = B.values()
    assert A == Matrix([[0, B12, B13], [-B13, -B23, 0]])
    assert rows == ['e1', 'e3']
    assert columns == ['e1', 'e2', 'e3']

    # Test for the matrix rep of the anti-commutator.
    A, rows, columns = expr_as_matrix(alg.acp, B, x)
    assert A == Matrix([[B23, -B13, B12]])
    assert rows == ['e123']
    assert columns == ['e1', 'e2', 'e3']

    # Test for the matrix rep of conjugation of the e3 plane.
    x = alg.vector(name='x', keys=('e3',))
    A, rows, columns = expr_as_matrix(alg.sw, B, x)
    assert A == Matrix([[2*B12*B23], [-2*B12*B13], [B12**2 - B13**2 - B23**2]])
    assert rows == ['e1', 'e2', 'e3']
    assert columns == ['e3']
