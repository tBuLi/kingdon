import pytest
from kingdon.polynomial import Polynomial, RationalPolynomial, compare


@pytest.mark.parametrize("a, b, res", [
    # Equal polynomials
    (Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 0),
    # a < b
    (Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), Polynomial([[1, 'x'], [2, 'y'], [4, 'z']]), -1),
    # a > b
    (Polynomial([[1, 'x'], [2, 'y'], [4, 'z']]), Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 1),
    # b undefined
    (Polynomial([[1, 'x'], [2, 'y'], [4, 'z']]), None, -1),
    # a undefined
    (None, Polynomial([[1, 'x'], [2, 'y'], [4, 'z']]), 1),
    # Both undefined
    (None, None, 1),
])
def test_compare(a, b, res):
    assert compare(a, b) == res


@pytest.mark.parametrize("a, b, res", [
    # Add single variable polynomials
    (Polynomial([[1, 'x']]), Polynomial([[2, 'x']]), Polynomial([[3, 'x']])),
    # Add multi-variable polynomials
    (Polynomial([[1, 'x', 'y'], [2, 'z']]), Polynomial([[3, 'x', 'y'], [4, 'z']]), Polynomial([[4, 'x', 'y'], [6, 'z']])),
    # Add maintaining sort order
    (Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), Polynomial([[4, 'x'], [5, 'y'], [6, 'z']]), Polynomial([[5, 'x'], [7, 'y'], [9, 'z']])),
    # Add with zero polynomials
    (Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 0, Polynomial([[1, 'x'], [2, 'y'], [3, 'z']])),
    # Add with negative coefficients
    (Polynomial([[1, 'x'], [-2, 'y'], [3, 'z']]), Polynomial([[-1, 'x'], [2, 'y'], [-3, 'z']]), 0),
    # Same denominator
    (RationalPolynomial([[1, 'x'], [2, 'y']], [[1, 'z']]), RationalPolynomial([[3, 'x'], [4, 'y']], [[1, 'z']]),
     RationalPolynomial([[4, 'x'], [6, 'y']], [[1, 'z']])),
    # Different denoms
    (RationalPolynomial([[1, 'x'], [2, 'y']], [[1, 'z']]), RationalPolynomial([[3, 'x'], [4, 'y']], [[2, 'z']]),
     RationalPolynomial([[5, 'x', 'z'], [8, 'y', 'z']], [[2, 'z', 'z']])),
    # Add with zero
    (RationalPolynomial([[1, 'x'], [2, 'y']], [[1, 'z']]), 0,
     RationalPolynomial([[1, 'x'], [2, 'y']], [[1, 'z']])),
    # With negative coefficients
    (RationalPolynomial([[-1, 'x'], [2, 'y']], [[1, 'z']]), RationalPolynomial([[1, 'x'], [-2, 'y']], [[1, 'z']]), 0),
])
def test_add(a, b, res):
    assert a + b == res


@pytest.mark.parametrize("a, b, res", [
    # single variable polynomials
    (Polynomial([[1, 'x']]), Polynomial([[2, 'x']]), Polynomial([[2, 'x', 'x']])),
    # multi-variable polynomials
    (Polynomial([[1, 'x', 'y'], [2, 'z']]), Polynomial([[3, 'x', 'y'], [4, 'z']]),
     Polynomial([[3, 'x', 'x', 'y', 'y'], [10, 'x', 'y', 'z'], [8, 'z', 'z']])),
    # with zero polynomials
    (Polynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 0, 0),
    # With negative coefficients
    (Polynomial([[1, 'x'], [-2, 'y'], [3, 'z']]), Polynomial([[-1, 'x'], [2, 'y'], [-3, 'z']]),
     Polynomial([[-1, 'x', 'x'], [4, 'x', 'y'], [-6, 'x', 'z'],[-4, 'y', 'y'], [12, 'y', 'z'], [-9, 'z', 'z']])),
    (RationalPolynomial([[1, 'x']]), RationalPolynomial([[2, 'x']]), RationalPolynomial([[2, 'x', 'x']])),
    (RationalPolynomial([[1, 'x', 'y'], [2, 'z']], [[1]]), RationalPolynomial([[3, 'x', 'y'], [4, 'z']], [[1]]),
     RationalPolynomial([[3, 'x', 'x', 'y', 'y'], [10, 'x', 'y', 'z'], [8, 'z', 'z']], [[1]])),
    (RationalPolynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 0, 0),
    (RationalPolynomial([[1, 'x'], [2, 'y'], [3, 'z']]), 1, RationalPolynomial([[1, 'x'], [2, 'y'], [3, 'z']])),
    (RationalPolynomial([[2, 'x', 'z']], [[1]]), RationalPolynomial([[1]], [[3, 'z']]),
     RationalPolynomial([[2, 'x',]], [[3]])),
])
def test_mul(a, b, res):
    assert a * b == res


@pytest.mark.parametrize("a, res", [
    # single variable polynomials
    (Polynomial([[1, 'x']]), Polynomial([[-1, 'x']])),
    (Polynomial([[1, 'x', 'y'], [2, 'z']]), Polynomial([[-1, 'x', 'y'], [-2, 'z']])),
    (Polynomial([]), Polynomial([])),
    (Polynomial([[-1, 'x'], [2, 'y'], [-3, 'z']]), Polynomial([[1, 'x'], [-2, 'y'], [3, 'z']])),
])
def test_neg(a, res):
    assert -a == res


@pytest.mark.parametrize("a, res", [
    # single variable polynomials
    (RationalPolynomial([[1, 'x']]), RationalPolynomial([[1]], [[1, 'x']])),
    (RationalPolynomial([[1, 'x', 'y'], [2, 'z']], [[1]]), RationalPolynomial([[1]], [[1, 'x', 'y'], [2, 'z']])),
])
def test_inverse(a, res):
    assert a.inv() == res


@pytest.mark.parametrize("a, res", [
    (RationalPolynomial([[1, 'x']]), RationalPolynomial([[1]])),
    (RationalPolynomial([[1, 'x','x'],[2, 'x', 'y'], [3, 'y', 'y', 'y']]), RationalPolynomial([[1]])),
])
def test_inverse_mul(a, res):
    assert a * a.inv() == res

@pytest.mark.parametrize("a, n, res", [
    (Polynomial([[2, 'x']]), 2, Polynomial([[4, 'x', 'x']])),
    (RationalPolynomial([[2, 'x']]), 2, RationalPolynomial([[4, 'x', 'x']])),
    (Polynomial([[2, 'x']]), 5, Polynomial([[2**5, 'x', 'x', 'x', 'x', 'x']])),
    (RationalPolynomial([[2, 'x']]), 5, RationalPolynomial([[2**5, 'x', 'x', 'x', 'x', 'x']])),
])
def test_pow(a, n, res):
    assert a ** n == res
