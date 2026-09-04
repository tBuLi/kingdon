import numpy as np
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


@pytest.mark.parametrize("poly, var, expected", [
    (Polynomial([[1, 'x']]), 'x', Polynomial([[1]])),  # d/dx(x) = 1
    (Polynomial([[1, 'x']]), Polynomial.fromname('x'), Polynomial([[1]])),  # d/dx(x) = 1
    (Polynomial([[2, 'x']]), 'x', Polynomial([[2]])),  # d/dx(2x) = 2
    (Polynomial([[1, 'x', 'x']]), 'x', Polynomial([[2, 'x']])),  # d/dx(x^2) = 2x
    (Polynomial([[1, 'x', 'x', 'x']]), 'x', Polynomial([[3, 'x', 'x']])),  # d/dx(x^3) = 3x^2
    (Polynomial([[1, 'x', 'x']]), Polynomial.fromname('x'), Polynomial([[2, 'x']])),  # d/dx(x^2) = 2x
    (Polynomial([[1, 'x', 'y']]), 'x', Polynomial([[1, 'y']])),  # d/dx(xy) = y
    (Polynomial([[1, 'x', 'y']]), 'y', Polynomial([[1, 'x']])),  # d/dy(xy) = x
    (Polynomial([[1, 'x', 'x', 'y']]), 'x', Polynomial([[2, 'x', 'y']])),  # d/dx(x^2*y) = 2xy
    (Polynomial([[1, 'x', 'x', 'y']]), 'y', Polynomial([[1, 'x', 'x']])),  # d/dy(x^2*y) = x^2
    (Polynomial([[1, 'x', 'y', 'z']]), 'x', Polynomial([[1, 'y', 'z']])),  # d/dx(xyz) = yz
    (Polynomial([[5]]), 'x', Polynomial([])),  # d/dx(5) = 0
    (Polynomial([]), 'x', Polynomial([])),  # d/dx(0) = 0
    (Polynomial([[1, 'x'], [1, 'y']]), 'x', Polynomial([[1]])),  # d/dx(x+y) = 1
    (Polynomial([[1, 'x'], [1, 'y'], [1, 'z']]), 'x', Polynomial([[1]])),  # d/dx(x+y+z) = 1
    (Polynomial([[1], [2, 'x'], [3, 'x', 'x']]), 'x', Polynomial([[2], [6, 'x']])),  # d/dx(3x^2+2x+1) = 6x+2
    (Polynomial([[1], [2, 'x'], [3, 'x', 'x']]), 'y', Polynomial([])),  # d/dy(3x^2+2x+1) = 0
    (Polynomial([[1, 'x', 'x', 'y'], [1, 'x', 'z']]), 'x', Polynomial([[2, 'x', 'y'], [1, 'z']])),  # d/dx(x^2*y+xz) = 2xy+z
    (Polynomial([[1, 'x', 'x'], [1, 'y', 'y']]), 'x', Polynomial([[2, 'x']])),  # d/dx(x^2+y^2) = 2x
    (Polynomial([[-3, 'x', 'x']]), 'x', Polynomial([[-6, 'x']])),  # d/dx(-3x^2) = -6x
    (Polynomial([[1, 'y', 'y']]), 'x', Polynomial([])),  # d/dx(y^2) = 0
])
def test_diff_polynomial(poly, var, expected):
    assert poly.diff(var) == expected


@pytest.mark.parametrize("rp, var, expected", [
    (RationalPolynomial([[1, 'x']], [[1, 'y']]), 'x',
     RationalPolynomial([[1, 'y']], [[1, 'y', 'y']])),  # d/dx(x/y) = 1/y
    (RationalPolynomial([[1, 'x']], [[1, 'y']]), 'y',
     RationalPolynomial([[-1, 'x']], [[1, 'y', 'y']])),  # d/dy(x/y) = -x/y^2
    (RationalPolynomial([[1]], [[1, 'x']]), 'x',
     RationalPolynomial([[-1]], [[1, 'x', 'x']])),  # d/dx(1/x) = -1/x^2
    (RationalPolynomial([[1, 'x', 'x']], [[1, 'y']]), 'x',
     RationalPolynomial([[2, 'x', 'y']], [[1, 'y', 'y']])),  # d/dx(x^2/y) = 2xy/y^2
    (RationalPolynomial([[1, 'x', 'x']], [[1, 'y']]), Polynomial.fromname('x'),
     RationalPolynomial([[2, 'x', 'y']], [[1, 'y', 'y']])),  # d/dx(x^2/y) = 2xy/y^2
    (RationalPolynomial([[1, 'x', 'x']]), 'x',
     RationalPolynomial([[2, 'x']], [[1]])),  # d/dx(x^2) = 2x
    (RationalPolynomial([[3]]), 'x',
     RationalPolynomial([])),  # d/dx(3) = 0
    (RationalPolynomial([[1, 'x']], [[1, 'y']]), 'z',
     RationalPolynomial([])),  # d/dz(x/y) = 0
    (RationalPolynomial([[1, 'x'], [1, 'y']], [[1, 'z']]), 'x',
     RationalPolynomial([[1, 'z']], [[1, 'z', 'z']])),  # d/dx((x+y)/z) = 1/z
    (RationalPolynomial([[1]], [[1, 'x', 'x']]), 'x',
     RationalPolynomial([[-2, 'x']], [[1, 'x', 'x', 'x', 'x']])),  # d/dx(1/x^2) = -2/x^3
])
def test_diff_rational_polynomial(rp, var, expected):
    assert rp.diff(var) == expected


@pytest.mark.parametrize("cls", [Polynomial, RationalPolynomial])
def test_no_arrays(cls):
    with pytest.raises(TypeError, match='ndarray'):
        cls(np.asarray(0.5))

    # Numbers are still fine, including the numpy scalars that are float subclasses.
    assert cls(0.5) == cls(np.float64(0.5))
