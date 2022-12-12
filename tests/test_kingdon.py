#!/usr/bin/env python

"""Tests for `kingdon` package."""
from dataclasses import replace

import pytest
import numpy as np

from sympy import Symbol, simplify, factor, expand, collect, sympify
from kingdon import Algebra, MultiVector, symbols

import timeit

@pytest.fixture
def pga1d():
    return Algebra(1, 0, 1)


@pytest.fixture
def pga2d():
    return Algebra(2, 0, 1)


@pytest.fixture
def pga3d():
    return Algebra(3, 0, 1)


@pytest.fixture
def vga2d():
    return Algebra(2)


@pytest.fixture
def vga11():
    return Algebra(1, 1)

@pytest.fixture
def R6():
    return Algebra(6)

@pytest.fixture
def sta():
    return Algebra(3, 1)

@pytest.fixture
def vga3d():
    return Algebra(3)


def test_MultiVector(pga1d):
    with pytest.raises(TypeError):
        # If starting from a sequence, it must be of length len(algebra)
        X = MultiVector(algebra=pga1d, vals=[1, 2])
    with pytest.raises(TypeError):
        # vals must be iterable (sequence)
        X = MultiVector(algebra=pga1d, values=2)
    with pytest.raises(TypeError):
        # No algebra provided
        X = MultiVector(name='X')
    with pytest.raises(KeyError):
        # Dict keys must be either (binary) numbers or canonical basis element strings.
        X = MultiVector(values={'a': 2, 'e12': 1}, algebra=pga1d)
    X = MultiVector(values={0: 2.2, 'e12': 1.2}, algebra=pga1d)
    assert dict(X.items()) == {0: 2.2, 3: 1.2}

def test_anticommutation(pga1d, vga11, vga2d):
    for alg in [pga1d, vga11, vga2d]:
        X = alg.multivector({1: 1})
        Y = alg.multivector({2: 1})
        assert X*Y == -Y*X

def test_gp(pga1d):
    # Multiply two multivectors
    X = MultiVector(values={'1': 2, 'e12': 3}, algebra=pga1d)
    Y = MultiVector(values={'1': 7, 'e12': 5}, algebra=pga1d)
    Z = X.gp(Y)
    assert dict(Z.items()) == {0: 2*7, 3: 2*5 + 3*7}

def test_cayley(pga1d, vga2d, vga11):
    assert pga1d.cayley == {('1', '1'): '1', ('1', 'e1'): 'e1', ('1', 'e12'): 'e12', ('1', 'e2'): 'e2',
                          ('e1', '1'): 'e1', ('e1', 'e1'): '1', ('e1', 'e12'): 'e2', ('e1', 'e2'): 'e12',
                          ('e12', '1'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e12'): '0', ('e12', 'e2'): '0',
                          ('e2', '1'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e12'): '0', ('e2', 'e2'): '0'}
    assert vga2d.cayley == {('1', '1'): '1', ('1', 'e1'): 'e1', ('1', 'e2'): 'e2', ('1', 'e12'): 'e12',
                          ('e1', '1'): 'e1', ('e1', 'e1'): '1', ('e1', 'e2'): 'e12', ('e1', 'e12'): 'e2',
                          ('e2', '1'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e2'): '1', ('e2', 'e12'): '-e1',
                          ('e12', '1'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e2'): 'e1', ('e12', 'e12'): '-1'}
    assert vga11.cayley == {('1', '1'): '1', ('1', 'e1'): 'e1', ('1', 'e2'): 'e2', ('1', 'e12'): 'e12',
                          ('e1', '1'): 'e1', ('e1', 'e1'): '1', ('e1', 'e2'): 'e12', ('e1', 'e12'): 'e2',
                          ('e2', '1'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e2'): '-1', ('e2', 'e12'): 'e1',
                          ('e12', '1'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e2'): '-e1', ('e12', 'e12'): '1'}

def test_purevector(pga1d):
    with pytest.raises(TypeError):
        # Grade needs to be specified.
        pga1d.purevector({1: 1, 2: 1})
    with pytest.raises(ValueError):
        # Grade must be valid, in this case no larger than 2.
        pga1d.purevector({1: 1, 2: 1}, grade=10)
    with pytest.raises(ValueError):
        # vals must be of the specified grade.
        pga1d.purevector({0: 1, 2: 1}, grade=1)
    x = pga1d.purevector({1: 1, 2: 1}, grade=1)
    assert isinstance(x, MultiVector)
    assert x.grades == (1,)

def test_broadcasting(vga2d):
    valsX = np.random.random((2, 5))
    valsY = np.random.random((2, 5))
    # Test if multiplication is correctly broadcast
    X = vga2d.vector(valsX)
    Y = vga2d.vector(valsY)
    Z = X * Y
    # test if the scalar and bivector part are what we expect.
    assert np.all(Z[0] == valsX[0] * valsY[0] + valsX[1] * valsY[1])
    assert np.all(Z[3] == valsX[0] * valsY[1] - valsX[1] * valsY[0])
    # Test multiplication by a scalar.
    Z = X * 3.0
    assert np.all(Z[1] == 3.0 * valsX[0])
    assert np.all(Z[2] == 3.0 * valsX[1])
    Z2 = 3.0 * X
    assert np.all(Z[1] == Z2[1]) and np.all(Z[2] == Z2[2])

    # Test broadcasting a rotor across a tensor-valued element
    R = vga2d.multivector({0: np.cos(np.pi / 3), 3: np.sin(np.pi / 3)})
    Z3 = R.conj(X)
    for i, xrow in enumerate(valsX.T):
        Rx = R.conj(vga2d.vector(xrow))
        assert Rx[1] == Z3[1][i]

def test_reverse(R6):
    X = R6.multivector(np.arange(0, 2 ** 6))
    Xrev = ~X
    assert X.grade((0, 1, 4, 5)) == Xrev.grade((0, 1, 4, 5))
    assert X.grade((2, 3, 6)) == - Xrev.grade((2, 3, 6))

def test_indexing(pga1d):
    # Test indexing of a mv with canonical and binary indices.
    X = pga1d.multivector({0: 2, 'e12': 3})
    assert X['1'] == 2 and X[3] == 3

def test_gp_symbolic(vga2d):
    u = vga2d.vector(name='u')
    u1, u2 = u[1], u[2]
    usq = u*u
    # Square of a vector should be purely scalar.
    assert usq[0] == u1**2 + u2**2
    assert len(usq) == 1
    assert 'e12' not in usq
    assert 0 in usq
    # Asking for an element that is not there always returns zero.
    # It does not raise a KeyError, because that might break people's code.
    assert usq['e12'] == 0

    # A bireflection should have both a scalar and bivector part however.
    v = vga2d.vector(name='v')
    v1, v2 = v[1], v[2]
    R = u*v
    assert R[0] == u1 * v1 + u2 * v2
    assert R[3] == u1 * v2 - u2 * v1

    # The norm of a bireflection is a scalar.
    Rnormsq = R*~R
    assert Rnormsq[0] == (u1*v1 + u2*v2)**2 - (-u1*v2 + u2*v1)*(u1*v2 - u2*v1)
    assert len(Rnormsq) == 1
    assert 'e12' not in Rnormsq
    assert 0 in Rnormsq
    assert Rnormsq['e12'] == 0

def test_conj_symbolic(vga2d):
    u = vga2d.vector(name='u')
    v = vga2d.vector(name='v')
    # Pure vector
    assert u.conj(v).grades == (1,)

def test_cp_symbolic(R6):
    b = R6.bivector(name='B')
    v = R6.vector(name='v')
    assert b.grades == (2,)
    assert v.grades == (1,)
    # Pure vector
    w = b.cp(v)
    assert w.grades == (1,)

def test_blades(vga2d):
    assert vga2d.blades['1'] == vga2d.multivector({'1': 1})
    assert vga2d.blades['e1'] == vga2d.multivector({'e1': 1})
    assert vga2d.blades['e2'] == vga2d.multivector({'e2': 1})
    assert vga2d.blades['e12'] == vga2d.multivector({'e12': 1})
    assert vga2d.blades['e12'] == vga2d.pss

def test_outer(sta):
    # Anticommutation of basis vectors.
    e1, e2 = sta.blades['e1'], sta.blades['e2']
    assert e1 ^ e2 == - e2 ^ e1

    # Test basis bivectors.
    e12, e23 = sta.blades['e12'], sta.blades['e23']
    assert not (e12 ^ e23)
    e12, e34 = sta.blades['e12'], sta.blades['e34']
    assert (e12 ^ e34) == (e34 ^ e12)

    # Non-simple bivector.
    B = sta.bivector(name='B')
    BwB = B ^ B
    assert BwB.grades == (4,)
    assert BwB[15] == 2*(B['e12']*B['e34'] - B['e13']*B['e24'] + B['e14']*B['e23'])

def test_alg_graded(vga2d):
    vga2d_graded = replace(vga2d, graded=True)
    assert vga2d != vga2d_graded
    u = vga2d_graded.multivector({'e1': 1})
    v = vga2d_graded.multivector({'e2': 3})
    print(u*v)


def test_inner_products(vga2d):
    a = vga2d.multivector(name='a')
    b = vga2d.multivector(name='b')

    bipa = b.ip(a)
    bspa = b.sp(a)
    blca = b.lc(a)
    brca = b.rc(a)

    # Inner product relation 2.11 from "The Inner Products of Geometric Algebra"
    assert bipa + bspa == blca + brca

    # Compare to output of GAmphetamine.js
    assert all([str(bipa[0]).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(bipa[1]).replace(' ', '') == 'a*b1+a1*b-a12*b2+a2*b12',
                str(bipa[2]).replace(' ', '') == 'a*b2-a1*b12+a12*b1+a2*b',
                str(bipa[3]).replace(' ', '') == 'a*b12+a12*b'])
    assert all([str(blca[0]).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(blca[1]).replace(' ', '') == 'a1*b-a12*b2',
                str(blca[2]).replace(' ', '') == 'a12*b1+a2*b',
                str(blca[3]).replace(' ', '') == 'a12*b'])
    assert all([str(brca[0]).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(brca[1]).replace(' ', '') == 'a*b1+a2*b12',
                str(brca[2]).replace(' ', '') == 'a*b2-a1*b12',
                str(brca[3]).replace(' ', '') == 'a*b12'])

def test_hodge_dual(pga2d, pga3d):
    x = pga2d.multivector(name='x')
    with pytest.raises(ZeroDivisionError):
        x.dual(kind='polarity')
    y = x.dual()
    # GAmphetamine.js output
    assert dict(y.items()) == {0: x[7], 1: x[6], 2: -x[5], 4: x[3], 3: x[4], 5: -x[2], 6: x[1], 7: x[0]}
    z = y.undual()
    assert x == z
    with pytest.raises(ValueError):
        x.dual('poincare')

    # Test hodge dual in 3DPGA
    x = pga3d.multivector(name='x')
    with pytest.raises(ZeroDivisionError):
        x.dual(kind='polarity')
    y = x.dual()
    # GAmphetamine.js output
    "x1234 - x234 e₁ + x134 e₂ - x124 e₃ + x123 e₄ + x34 e₁₂ - x24 e₁₃ + x23 e₁₄ + x14 e₂₃ - x13 e₂₄ + x12 e₃₄ " \
    "- x4 e₁₂₃ + x3 e₁₂₄ - x2 e₁₃₄ + x1 e₂₃₄ + x e₁₂₃₄"
    assert dict(y.items()) == {
        0b0000: x[0b1111],
        0b0001: -x[0b1110], 0b0010: x[0b1101], 0b0100: -x[0b1011], 0b1000: x[0b0111],
        0b0011: x[0b1100], 0b0101: -x[0b1010], 0b1001: x[0b0110], 0b0110: x[0b1001], 0b1010: -x[0b0101], 0b1100: x[0b0011],
        0b0111: -x[0b1000], 0b1011: x[0b0100], 0b1101: -x[0b0010], 0b1110: x[0b0001],
        0b1111: x[0]
    }
    z = y.undual()
    assert z == x

def test_regressive(pga3d):
    """ Test the regressive product of full mvs in 3DPGA against the known result from GAmphetamine.js"""
    xvals = symbols(','.join(f'x{i}' for i in range(1, len(pga3d) + 1)))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = xvals
    x = pga3d.multivector({k: xvals[i] for i, k in enumerate(pga3d.canon2bin)})
    yvals = symbols(','.join(f'y{i}' for i in range(1, len(pga3d) + 1)))
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 = yvals
    y = pga3d.multivector({k: yvals[i] for i, k in enumerate(pga3d.canon2bin)})

    # Known output from GAmphetamine.js
    known_vals = {
        "1": (x1*y16-x10*y7+x11*y6+x12*y5-x13*y4+x14*y3-x15*y2+x16*y1+x2*y15-x3*y14+x4*y13-x5*y12+x6*y11-x7*y10+x8*y9+x9*y8),
        "e1": (x12*y8-x13*y7+x14*y6+x16*y2+x2*y16+x6*y14-x7*y13+x8*y12),
        "e2": (x10*y12+x12*y10-x13*y9+x15*y6+x16*y3+x3*y16+x6*y15-x9*y13),
        "e3": (x11*y12+x12*y11-x14*y9+x15*y7+x16*y4+x4*y16+x7*y15-x9*y14),
        "e4": (-x10*y14+x11*y13+x13*y11-x14*y10+x15*y8+x16*y5+x5*y16+x8*y15),
        "e12": (x12*y13-x13*y12+x16*y6+x6*y16),
        "e13": (x12*y14-x14*y12+x16*y7+x7*y16),
        "e14": (x13*y14-x14*y13+x16*y8+x8*y16),
        "e23": (x12*y15-x15*y12+x16*y9+x9*y16),
        "e24": (x10*y16+x13*y15-x15*y13+x16*y10),
        "e34": (x11*y16+x14*y15-x15*y14+x16*y11),
        "e123": (x12*y16+x16*y12),
        "e124": (x13*y16+x16*y13),
        "e134": (x14*y16+x16*y14),
        "e234": (x15*y16+x16*y15),
        "e1234": (x16*y16)
    }
    known = pga3d.multivector(known_vals)
    x_regr_y = x & y
    for i in range(len(pga3d)):
        assert x_regr_y[i] == known[i]


def test_projection(pga3d):
    x1, x2, x3 = symbols('x1, x2, x3')
    x = pga3d.trivector([x1, x2, x3, 1])
    y1, y2, y3, y4 = symbols('y1, y2, y3, y4')
    y = pga3d.vector([y1, y2, y3, y4])
    # project the plane y onto the point x
    z = y @ x
    assert z.grades == (1,)
    # project the point x onto the plane y
    z = x @ y
    assert z.grades == (3,)


def test_inverse_div(pga2d):
    u = pga2d.multivector(name='u')
    # Multiply by inverse results in a scalar exp, which numerically evaluates to 1.
    res = u*u.inv()
    assert res.grades == (0,)
    # All the null elements will have disappeared from the output,
    # so only four values left to provide.
    u_vals = np.random.random(4)
    assert res(*u_vals)[0] == pytest.approx(1.0)
    # Division by self is truly the scalar 1.
    res = u / u
    assert res.grades == (0,)
    assert res[0] == 1


def test_mixed_symbolic(vga2d):
    x = vga2d.evenmv({0: 2.2, 3: 's'})
    assert x[3] == Symbol('s')
    assert x[0] == 2.2
    assert x.issymbolic


def test_evenmultivector(R6):
    x = R6.evenmv(name='x')
    assert x.grades == (0, 2, 4, 6)


def test_oddmultivector(R6):
    x = R6.oddmv(name='x')
    assert x.grades == (1, 3, 5)


def test_matrixreps(vga2d):
    x = vga2d.multivector(name='x')
    xmat = x.asmatrix()
    xprime = MultiVector.frommatrix(vga2d, matrix=xmat)
    assert np.all(x.values() == xprime.values())
    assert x.keys() == xprime.keys()


def test_fromkeysvalues():
    alg = Algebra(2, numba=False)
    xvals = symbols('x x1 x2 x12')
    xkeys = tuple(range(4))
    x = alg.multivector(keys=xkeys, values=xvals)

    assert x._values is xvals
    assert x._keys is xkeys

    # We use sympify, so string that look like equations are also allowed
    y = alg.multivector(['a*b+c', '-15*c'], grades=(1,))
    assert y[1] == Symbol('a')*Symbol('b') + Symbol('c')
    assert y[2] == -15 * Symbol('c')

    yvals = symbols('y y1 y2 y12')
    with pytest.raises(TypeError):
        y = alg.multivector(yvals, xkeys[:3])
    y = alg.multivector(yvals)
    assert y._values is yvals
    assert y._keys == xkeys

    xy = x * y
    assert xy[0] == sympify("(x*y+x1*y1-x12*y12+x2*y2)")
    assert xy['e1'] == sympify("(x*y1+x1*y+x12*y2-x2*y12)")
    assert xy[2] == sympify("(x*y2+x1*y12-x12*y1+x2*y)")
    assert xy['e12'] == sympify("(x*y12+x1*y2+x12*y-x2*y1)")

def test_commutator():
    alg = Algebra(2, 1, 1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    xcpy = x.cp(y)
    xcpy_expected = ((x*y)-(y*x)) / 2
    for i in range(len(alg)):
        assert xcpy[i] - xcpy_expected[i] == 0

def test_anticommutator():
    alg = Algebra(2, 1, 1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    xacpy = x.acp(y)
    xacpy_expected = ((x*y)+(y*x)) / 2
    for i in range(len(alg)):
        assert xacpy[i] - xacpy_expected[i] == 0

def test_conjugation():
    alg = Algebra(1, 1, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')

    xconjy_expected = x*y*(~x)
    xconjy = x.conj(y)
    for i in range(len(alg)):
        assert expand(xconjy[i]) == expand(xconjy_expected[i])

def test_projection():
    alg = Algebra(1, 1, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')

    xconjy_expected = (x | y) * ~y
    xconjy = x.proj(y)
    for i in range(len(alg)):
        assert expand(xconjy[i]) == expand(xconjy_expected[i])


def test_outerexp(R6):
    B = R6.bivector(name='B')
    LB = B.outerexp()
    LB_exact = 1 + B + (B ^ B) / 2 + (B ^ B ^ B) / 6

    diff = LB - LB_exact
    for val in diff.values():
        assert simplify(val) == 0

    v = R6.vector(name='v')
    Lv = v.outerexp()
    Lv_exact = 1 + v
    diff = Lv - Lv_exact
    for val in diff.values():
        assert val == 0

def test_outertrig(R6):
    alg = Algebra(6)
    B = alg.bivector(name='B', keys=(0b110000, 0b1100, 0b11))
    sB = B.outersin()
    cB = B.outercos()

    sB_exact = B + (B ^ B ^ B) / sympify(6)
    cB_exact = sympify(1) + (B ^ B) / sympify(2)

    for diff in [sB - sB_exact, cB - cB_exact]:
        assert all(v == 0 for v in diff.values())


def test_indexing():
    alg = Algebra(4)
    nrows = 3
    bvals = np.random.random((len(alg.indices_for_grade[2]), nrows))
    B = alg.bivector(bvals)
    np.testing.assert_allclose(B[3, 2:4], bvals[0, 2:4])
    np.testing.assert_allclose(B[3, 2], bvals[0, 2])
    np.testing.assert_allclose(B[3, :], bvals[0, :])
    np.testing.assert_allclose(B[:, 0], bvals[:, 0])

    #TODO: same tests but without using a numpy array
    bvals = tuple(np.random.random(nrows) for _ in range(len(alg.indices_for_grade[2])))
    B = alg.bivector(bvals)
    np.testing.assert_allclose(B[3, 2:4], bvals[0][2:4])
    np.testing.assert_allclose(B[3, 2], bvals[0][2])
    np.testing.assert_allclose(B[3, :], bvals[0][:])
    np.testing.assert_allclose(B[:, 0], bvals[:][0])

def test_normalization(pga3d):
    vvals = np.random.random(len(pga3d.indices_for_grade[1]))
    v = pga3d.vector(vvals).normalized()
    assert (v*v)[0] == pytest.approx(1.0)
    np.testing.assert_allclose((v*v)[0], 1.0)

    bvals = np.random.random(len(pga3d.indices_for_grade[2]))
    with pytest.raises(NotImplementedError):
        B = pga3d.bivector(bvals).normalized()
