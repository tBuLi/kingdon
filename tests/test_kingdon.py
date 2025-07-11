#!/usr/bin/env python

"""Tests for `kingdon` package."""
import copy
import itertools
import operator
from dataclasses import replace

import pytest
import numpy as np

from sympy import Symbol, expand, sympify, cos, sin, sinc
from kingdon import Algebra, MultiVector, symbols
from kingdon.operator_dict import UnaryOperatorDict

import timeit

@pytest.fixture
def pga1d():
    return Algebra(signature=np.array([1, 0]), start_index=1)


@pytest.fixture
def pga2d():
    return Algebra(signature=np.array([1, 1, 0]), start_index=1)


@pytest.fixture
def pga3d():
    return Algebra(signature=np.array([1, 1, 1, 0]), start_index=1)


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
        X = MultiVector(algebra=pga1d, values=[1, 2])
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
    # Multiply two multivectors. Also tests two different MV creation API's
    X = MultiVector(values={'e': 2, 'e12': 3}, algebra=pga1d)
    Y = MultiVector(e=7, e12=5, algebra=pga1d)
    Z = X * Y
    assert dict(Z.items()) == {0: 2*7, 3: 2*5 + 3*7}

def test_cayley(pga1d, vga2d, vga11):
    assert pga1d.cayley == {('e', 'e'): 'e', ('e', 'e1'): 'e1', ('e', 'e12'): 'e12', ('e', 'e2'): 'e2',
                          ('e1', 'e'): 'e1', ('e1', 'e1'): 'e', ('e1', 'e12'): 'e2', ('e1', 'e2'): 'e12',
                          ('e12', 'e'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e12'): '0', ('e12', 'e2'): '0',
                          ('e2', 'e'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e12'): '0', ('e2', 'e2'): '0'}
    assert vga2d.cayley == {('e', 'e'): 'e', ('e', 'e1'): 'e1', ('e', 'e2'): 'e2', ('e', 'e12'): 'e12',
                          ('e1', 'e'): 'e1', ('e1', 'e1'): 'e', ('e1', 'e2'): 'e12', ('e1', 'e12'): 'e2',
                          ('e2', 'e'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e2'): 'e', ('e2', 'e12'): '-e1',
                          ('e12', 'e'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e2'): 'e1', ('e12', 'e12'): '-e'}
    assert vga11.cayley == {('e', 'e'): 'e', ('e', 'e1'): 'e1', ('e', 'e2'): 'e2', ('e', 'e12'): 'e12',
                          ('e1', 'e'): 'e1', ('e1', 'e1'): 'e', ('e1', 'e2'): 'e12', ('e1', 'e12'): 'e2',
                          ('e2', 'e'): 'e2', ('e2', 'e1'): '-e12', ('e2', 'e2'): '-e', ('e2', 'e12'): 'e1',
                          ('e12', 'e'): 'e12', ('e12', 'e1'): '-e2', ('e12', 'e2'): '-e1', ('e12', 'e12'): 'e'}

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
    assert np.all(Z.e == valsX[0] * valsY[0] + valsX[1] * valsY[1])
    assert np.all(Z.e12 == valsX[0] * valsY[1] - valsX[1] * valsY[0])
    # Test multiplication by a scalar.
    Z = X * 3.0
    assert np.all(Z.e1 == 3.0 * valsX[0])
    assert np.all(Z.e2 == 3.0 * valsX[1])
    Z2 = 3.0 * X
    assert np.all(Z.e1 == Z2.e1) and np.all(Z.e2 == Z2.e2)

    # Test broadcasting a rotor across a tensor-valued element
    R = vga2d.multivector({0: np.cos(np.pi / 3), 3: np.sin(np.pi / 3)})
    Z3 = R.sw(X)
    for i, xrow in enumerate(valsX.T):
        Rx = R.sw(vga2d.vector(xrow))
        assert Rx.e1 == Z3.e1[i]

def test_reverse(R6):
    X = R6.multivector(np.arange(0, 2 ** 6))
    Xrev = ~X
    assert X.grade((0, 1, 4, 5)) == Xrev.grade((0, 1, 4, 5))
    assert X.grade((2, 3, 6)) == - Xrev.grade((2, 3, 6))

def test_getattr_setattr(pga1d):
    X = pga1d.multivector({0: 2, 'e12': 3})
    assert X.e == 2 and X.e12 == 3
    assert X.e1 == 0 and X.e2 == 0
    # Asking for a valid basis blade outside of the algebra should also return 0
    assert X.e3 == 0
    with pytest.raises(AttributeError):
        X.someattrthatdoesntexist

    X.e = 3
    assert X.e == 3
    with pytest.raises(TypeError):
        X.e1 = 5
    assert dict(X.items()) == {0: 3, 3: 3}

    with pytest.raises(TypeError):
        del X.e

def test_gp_symbolic(vga2d):
    u = vga2d.vector(name='u')
    u1, u2 = u.e1, u.e2
    usq = u*u
    # Square of a vector should be purely scalar.
    assert usq.e == u1**2 + u2**2
    assert len(usq) == 1
    assert 'e12' not in usq
    assert 0 in usq
    # Asking for an element that is not there always returns zero.
    # It does not raise a KeyError, because that might break people's code.
    assert usq.e12 == 0

    # A bireflection should have both a scalar and bivector part however.
    v = vga2d.vector(name='v')
    v1, v2 = v.e1, v.e2
    R = u*v
    assert R.e == u1 * v1 + u2 * v2
    assert R.e12 == u1 * v2 - u2 * v1

    # The norm of a bireflection is a scalar.
    Rnormsq = R*~R
    assert expand(Rnormsq.e - ((u1*v1 + u2*v2)**2 + (u1*v2 - u2*v1)**2)) == 0
    assert len(Rnormsq) == 1
    assert 'e12' not in Rnormsq
    assert 0 in Rnormsq
    assert Rnormsq.e12 == 0

def test_sw_symbolic(vga2d):
    u = vga2d.vector(name='u')
    v = vga2d.vector(name='v')
    # Pure vector
    assert u.sw(v).grades == (1,)

def test_cp_symbolic(R6):
    b = R6.bivector(name='B')
    v = R6.vector(name='v')
    assert b.grades == (2,)
    assert v.grades == (1,)
    # Pure vector
    w = b.cp(v)
    assert w.grades == (1,)

def test_norm_euler():
    alg = Algebra(2)
    e, e12 = alg.blades['e'], alg.blades.e12
    t = Symbol('t')
    R = cos(t) * e + sin(t) * e12
    Rnormsq = R.normsq()
    assert Rnormsq.grades == (0,)
    assert Rnormsq.values()[0] == 1


def test_blades(vga2d):
    assert vga2d.blades['e'] == vga2d.multivector({'e': 1})
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
    assert BwB.e1234 == 2*(B.e12*B.e34 - B.e13*B.e24 + B.e14*B.e23)

def test_alg_graded(vga2d):
    vga2d_graded = replace(vga2d, graded=True)
    assert vga2d != vga2d_graded
    u = vga2d_graded.vector([1, 2])
    v = vga2d_graded.vector([0, 3])
    R = u * v
    assert R.grades == (0, 2)
    assert R.e == 6
    assert R.e1 == 0
    assert R.e2 == 0
    assert R.e12 == 3


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
    assert all([str(bipa.e).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(bipa.e1).replace(' ', '') == 'a*b1+a1*b-a12*b2+a2*b12',
                str(bipa.e2).replace(' ', '') == 'a*b2-a1*b12+a12*b1+a2*b',
                str(bipa.e12).replace(' ', '') == 'a*b12+a12*b'])
    assert all([str(blca.e).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(blca.e1).replace(' ', '') == 'a1*b-a12*b2',
                str(blca.e2).replace(' ', '') == 'a12*b1+a2*b',
                str(blca.e12).replace(' ', '') == 'a12*b'])
    assert all([str(brca.e).replace(' ', '') == 'a*b+a1*b1-a12*b12+a2*b2',
                str(brca.e1).replace(' ', '') == 'a*b1+a2*b12',
                str(brca.e2).replace(' ', '') == 'a*b2-a1*b12',
                str(brca.e12).replace(' ', '') == 'a*b12'])

def test_hodge_dual(pga2d, pga3d):
    x = pga2d.multivector(name='x')
    with pytest.raises(ZeroDivisionError):
        x.dual(kind='polarity')
    y = x.dual()
    # GAmphetamine.js output
    assert dict(y.items()) == {0: x.e123, 1: x.e23, 2: -x.e13, 4: x.e12, 3: x.e3, 5: -x.e2, 6: x.e1, 7: x.e}
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
        0b0000: x.e1234,
        0b0001: -x.e234, 0b0010: x.e134, 0b0100: -x.e124, 0b1000: x.e123,
        0b0011: x.e34, 0b0101: -x.e24, 0b1001: x.e23, 0b0110: x.e14, 0b1010: -x.e13, 0b1100: x.e12,
        0b0111: -x.e4, 0b1011: x.e3, 0b1101: -x.e2, 0b1110: x.e1,
        0b1111: x.e
    }
    z = y.undual()
    assert z == x

def test_polarity():
    alg = Algebra(4, 1)
    x = alg.multivector(name='x')
    xdual = x.polarity()
    assert not (xdual - x * alg.pss.inv()).values()
    assert not (xdual.unpolarity() - x).values()

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
        "e": (x1*y16-x10*y7+x11*y6+x12*y5-x13*y4+x14*y3-x15*y2+x16*y1+x2*y15-x3*y14+x4*y13-x5*y12+x6*y11-x7*y10+x8*y9+x9*y8),
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
    assert x_regr_y == known


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


def test_inv_div(pga2d):
    u = pga2d.multivector(name='u')
    # Multiply by inverse results in a scalar exp, which numerically evaluates to 1.
    res = u*u.inv()
    # All the null elements will have disappeared from the output,
    # so only four values left to provide.
    u_vals = np.random.random(4)
    assert res(*u_vals).e == pytest.approx(1.0)
    assert res.grades == (0,)
    assert res.e == 1
    # Division by self is truly the scalar 1.
    res = u / u
    assert res(*u_vals).e == pytest.approx(1.0)
    assert res.grades == (0,)
    assert res.e == 1

def test_hitzer_inv():
    from kingdon.polynomial import RationalPolynomial
    for d in range(5): # The d=5 case is excluded because the test is too slow.
        alg = Algebra(d)
        x = alg.multivector(name='x', symbolcls=RationalPolynomial.fromname)
        assert x * x.inv() == alg.blades.e


def test_mixed_symbolic(vga2d):
    x = vga2d.evenmv(e=2.2, e12='s')
    assert x.e12 == Symbol('s')
    assert x.e == 2.2
    assert x.issymbolic


def test_evenmultivector(R6):
    x = R6.evenmv(name='x')
    assert x.grades == (0, 2, 4, 6)


def test_oddmultivector(R6):
    x = R6.oddmv(name='x')
    assert x.grades == (1, 3, 5)


def test_namedmv(R6):
    keys = (1, 3, 17)
    x = R6.multivector(name='x', keys=keys)
    assert x.keys() == keys
    named_keys = ('e1', 'e12', 'e15')
    y = R6.multivector(name='x', keys=named_keys)
    assert x.keys() == keys


def test_fromkeysvalues():
    alg = Algebra(2)
    xvals = symbols('x x1 x2 x12')
    xkeys = tuple(range(4))
    x = alg.multivector(keys=xkeys, values=xvals)

    assert x._values is xvals
    assert x._keys is xkeys

    # We use sympify, so string that look like equations are also allowed
    y = alg.multivector(['a*b+c', '-15*c'], grades=(1,))
    assert y.e1 == Symbol('a')*Symbol('b') + Symbol('c')
    assert y.e2 == -15 * Symbol('c')

    yvals = symbols('y y1 y2 y12')
    with pytest.raises(TypeError):
        y = alg.multivector(yvals, xkeys[:3])
    y = alg.multivector(yvals)
    assert y._values is yvals
    assert y._keys == xkeys

    xy = x * y
    assert xy.e == sympify("(x*y+x1*y1-x12*y12+x2*y2)")
    assert xy.e1 == sympify("(x*y1+x1*y+x12*y2-x2*y12)")
    assert xy.e2 == sympify("(x*y2+x1*y12-x12*y1+x2*y)")
    assert xy.e12 == sympify("(x*y12+x1*y2+x12*y-x2*y1)")

def test_commutator():
    alg = Algebra(2, 1, 1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    xcpy = x.cp(y)
    xcpy_expected = ((x*y)-(y*x)) / 2
    diff = xcpy_expected - xcpy
    assert not diff

def test_anticommutator():
    alg = Algebra(2, 1, 1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    xacpy = x.acp(y)
    xacpy_expected = ((x*y)+(y*x)) / 2
    diff = xacpy_expected - xacpy
    assert not diff


def test_conjugation():
    alg = Algebra(1, 1, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')

    # Check if the built-in conjugation formula is what we expect it to be.
    xconjy_expected = x * y * ~x
    xconjy = x >> y
    diff = (xconjy_expected - xconjy)
    assert not diff


def test_projection():
    alg = Algebra(1, 1, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')

    xprojy_expected = (x | y) * ~y
    xprojy = x @ y
    diff = (xprojy_expected - xprojy)
    assert not diff


def test_outerexp(R6):
    B = R6.bivector(name='B')
    LB = B.outerexp()
    LB_exact = 1 + B + (B ^ B) / 2 + (B ^ B ^ B) / 6

    diff = LB - LB_exact
    assert not diff

    v = R6.vector(name='v')
    Lv = v.outerexp()
    Lv_exact = 1 + v
    diff = Lv - Lv_exact
    assert not diff

def test_outertrig(R6):
    alg = Algebra(6)
    B = alg.bivector(name='B', keys=(0b110000, 0b1100, 0b11))
    sB = B.outersin()
    cB = B.outercos()

    sB_exact = B + (B ^ B ^ B) / sympify(6)
    cB_exact = sympify(1) + (B ^ B) / sympify(2)

    for diff in [sB - sB_exact, cB - cB_exact]:
        assert not diff


def test_multidimensional_indexing():
    alg = Algebra(4)
    nrows = 3
    ncolumns = 4
    shape = (len(tuple(alg.indices_for_grade(2))), nrows, ncolumns)
    bvals = np.random.random(shape)
    B = alg.bivector(bvals)
    np.testing.assert_allclose(B[2:4].e12, bvals[0, 2:4])
    np.testing.assert_allclose(B[2].e12, bvals[0, 2])
    np.testing.assert_allclose(B[:].e12, bvals[0, :])
    np.testing.assert_allclose(B[0].values(), bvals[:, 0])
    np.testing.assert_allclose(B[2:4, 0].e12, bvals[0, 2:4, 0])
    # Same tests but without using a numpy array, instead use a tuple of sub np.ndarray.
    B = alg.bivector(tuple(bvals))
    np.testing.assert_allclose(B[2:4].e12, bvals[0, 2:4])
    np.testing.assert_allclose(B[2].e12, bvals[0, 2])
    np.testing.assert_allclose(B[:].e12, bvals[0, :])
    np.testing.assert_allclose(B[0].values(), bvals[:, 0])
    np.testing.assert_allclose(B[2:4, 0].e12, bvals[0, 2:4, 0])
    np.testing.assert_allclose(B[0, 0].values(), bvals[:, 0, 0])


def test_sqrt():
    alg = Algebra(3, 0, 1)
    uvals = np.random.random(4)
    vvals = np.random.random(4)
    u = alg.vector(uvals).normalized()
    v = alg.vector(vvals).normalized()
    R = u * v

    Rsqrt = R.sqrt()
    diff = Rsqrt*Rsqrt - R
    np.testing.assert_almost_equal(diff.values(), np.zeros(len(alg) // 2))

    Rsqrt_direct = (1 + R).normalized()
    diff = Rsqrt - Rsqrt_direct
    np.testing.assert_almost_equal(diff.values(), np.zeros(len(alg) // 2))


def test_clifford_involutions():
    alg = Algebra(8)
    x = alg.multivector(name='x')
    assert (x - x.reverse()).grades == (2, 3, 6, 7)
    assert (x - x.involute()).grades == (1, 3, 5, 7)
    assert (x - x.conjugate()).grades == (1, 2, 5, 6)
    assert (x.conjugate() == x.reverse().involute())


def test_normalization(pga3d):
    vvals = np.random.random(len(tuple(pga3d.indices_for_grade(1))))
    v = pga3d.vector(vvals).normalized()
    assert (v*v).e == pytest.approx(1.0)

    # Normalizing a non-simple bivector makes it simple!
    bvals = np.random.random(len(tuple(pga3d.indices_for_grade(2))))
    B = pga3d.bivector(bvals)
    Bnormalized = B.normalized()
    assert Bnormalized.normsq().e == pytest.approx(1.0)
    assert Bnormalized.normsq().e1234 == pytest.approx(0.0)


def test_itermv():
    alg = Algebra(4)
    nrows = 3
    shape = (len(tuple(alg.indices_for_grade(2))), nrows)
    bvals = np.random.random(shape)
    B = alg.bivector(bvals)
    for i, b in enumerate(B.itermv()):
        np.testing.assert_allclose(b.values(), bvals[:, i])
    assert i + 1 == nrows


def test_fromsignature():
    alg = Algebra(signature=[0, -1, 1, 1])
    assert alg.start_index == 0
    assert isinstance(alg.signature, np.ndarray)
    assert np.all(alg.signature == [0, -1, 1, 1])
    assert (alg.p, alg.q, alg.r) == (2, 1, 1)


def test_start_index():
    pga2d = Algebra(signature=[0, 1, 1], start_index=0)
    alg = Algebra(signature=[0, 1, 1], start_index=1)
    for ei, fi in zip(pga2d.blades.values(), alg.blades.values()):
        assert fi**2 == ei**2


def test_asfullmv():
    alg = Algebra(2, 0, 1)
    xvals = np.random.random(3)
    x = alg.vector(xvals)
    # Manually make the expected dense x
    x_densevals = np.zeros(len(alg))
    x_densevals[np.array([1, 2, 4])] = xvals
    x_dense = alg.multivector(x_densevals, keys=tuple(range(8)))
    # Compare to asfullmv method.
    y = x.asfullmv(canonical=False)
    assert y.keys() == x_dense.keys()
    np.testing.assert_equal(y.values(), x_dense.values())

    # Manually make the expected dense x in canonical ordering
    x_densevals = np.zeros(len(alg))
    x_densevals[np.array([1, 2, 3])] = xvals
    # Compare to asfullmv method.
    y = x.asfullmv()
    np.testing.assert_equal(y.values(), x_densevals)


def test_type():
    alg = Algebra(3)
    keys = (0b000, 0b100, 0b101, 0b111)
    x = alg.multivector(name='x', keys=keys)
    assert x.type_number == 0b10101001


def test_graded():
    alg = Algebra(2, 0, 1, graded=True)

    for b in alg.blades.values():
        assert len(b.grades) == 1
        assert b.keys() == tuple(alg.indices_for_grades(b.grades))

    with pytest.raises(ValueError):
        # In graded mode, the keys have to be correct.
        x = alg.multivector(name='x', keys=(1,))


def test_blade_dict():
    alg = Algebra(2)
    assert not alg.blades.lazy
    assert len(alg.blades) == len(alg)
    locals().update(**alg.blades)

    alg = Algebra(2, graded=True)
    assert not alg.blades.lazy
    assert len(alg.blades) == len(alg)
    assert len(alg.blades['e1']) == 2

    # In algebras larger than 6, lazy is the default.
    alg = Algebra(7)
    assert alg.blades.lazy
    assert len(alg.blades) == 1  # PSS is calculated by default
    assert len(alg.blades['e12']) == 1
    assert len(alg.blades) == 2

    alg = Algebra(7, graded=True)
    assert alg.blades.lazy
    assert len(alg.blades) == 1  # PSS is calculated by default
    assert len(alg.blades['e12']) == len(tuple(alg.indices_for_grade(2)))
    assert len(alg.blades) == 2


def test_numregister_operator_existence():
    """ Test if all battery-included GA operators can be used in custum functions."""
    alg = Algebra(2, 0, 0)
    uvals = np.random.random(len(alg))
    vvals = np.random.random(len(alg))
    u = alg.multivector(uvals).grade((0, 2))
    v = alg.multivector(vvals)

    operators = alg.registry.copy()
    for op_name, op_dict in operators.items():
        if isinstance(op_dict, UnaryOperatorDict):
            def myfunc(x):
                return getattr(x, op_name)()

            myfunc_compiled = alg.register(myfunc)
            assert myfunc_compiled(u) == myfunc(u)

        else:
            def myfunc(x, y):
                return getattr(x, op_name)(y)
            myfunc_compiled = alg.register(myfunc)
            assert myfunc_compiled(u, v) == myfunc(u, v)


def test_numregister_basics():
    alg = Algebra(3, 0, 1)
    uvals = np.random.random(len(alg))
    vvals = np.random.random(len(alg))
    u = alg.multivector(uvals)
    v = alg.multivector(vvals)

    @alg.register
    def square(x):
        return x * x

    @alg.register
    def double(x):
        return 2 * x

    @alg.register
    def add(x, y):
        return x + y

    @alg.register
    def grade_select(x):
        return x.grade(1, 2)

    # Test if we can nest registered expressions.
    @alg.register
    def coupled(u, v):
        uv = add(u, v)
        return square(uv) + double(u)

    assert square(u) == square.codegen(u)
    assert double(u) == double.codegen(u)
    assert add(u, v) == add.codegen(u, v)
    assert grade_select(u) == u.grade(1, 2)
    assert coupled(u, v) == (u + v)**2 + 2 * u


def test_symregister_basics():
    alg = Algebra(3, 0, 1)
    u = alg.multivector(name='u')
    v = alg.multivector(name='v')

    @alg.register(symbolic=True)
    def square(x):
        return x * x

    @alg.register(symbolic=True)
    def double(x):
        return 2 * x

    @alg.register(symbolic=True)
    def add(x, y):
        return x + y

    @alg.register(symbolic=True)
    def grade_select(x):
        return x.grade((1, 2))

    # Test if we can nest registered expressions.
    @alg.register(symbolic=True)
    def coupled(u, v):
        uv = add(u, v)
        return square(uv) + double(u)

    assert square(u) == square.codegen(u)
    assert double(u) == double.codegen(u)
    assert add(u, v) == add.codegen(u, v)
    assert grade_select(u) == u.grade((1, 2))
    assert coupled(u, v) == (u + v) ** 2 + 2 * u


def test_25():
    from kingdon import Algebra
    alg = Algebra(3, 0, 1)
    e0 = alg.blades['e0']
    e1 = alg.blades['e1']
    e2 = alg.blades['e2']
    e02 = alg.blades['e02']
    e12 = alg.blades['e12']

    x = e12
    y = (0 * e2 + e0).dual()
    z = e1.dual()
    ans = ((x * y) | z)
    assert ans == e02

def test_value_31():
    alg = Algebra(2)
    B = alg.bivector(name='B')
    res = 2 * (B ^ B)
    # res is not just zero, but an empty mv.
    empty = alg.multivector()
    assert res == empty
    zero = alg.multivector(e=0)
    assert res != zero


def test_reciprocal_frame():
    alg = Algebra(1, 3)
    for (i, ei), (j, Ej) in itertools.product(enumerate(alg.frame), enumerate(alg.reciprocal_frame)):
        if i == j:
            assert (ei | Ej).e == 1
        else:
            assert (ei | Ej).e == 0

def test_call_mv():
    alg = Algebra(3, 0, 1)
    u = alg.vector(name='u')
    usq = u * u
    res = usq(u1=np.cos(np.pi / 3), u2=np.sin(np.pi / 3), u3=0)
    assert pytest.approx(1.0) == res.e

def test_setitem():
    alg = Algebra(2, 0, 1)
    l = 6
    d = 3 / l
    point_vals = np.zeros((alg.d, l + 1))
    point_vals[0] = 1
    point_vals[1] = np.arange(l + 1) * d - 1.5
    points = alg.vector(point_vals).dual()
    points[-1] = points[-2]
    assert points[-1] == points[-2]

def test_mv_times_func():
    """If a mv is binaried with a function, we simply call it until it returns a multivector. """
    alg = Algebra(2, 0, 1)  # Smallest non-Abelian algebra, that property is important.
    x = alg.multivector(name='x')
    y = alg.multivector(name='x')
    yfunc = lambda: lambda: y
    # See if binary operators have been overloaded correctly!
    assert x + y == x + yfunc
    assert y + x == yfunc + x
    assert x - y == x - yfunc
    assert y - x == yfunc - x
    assert x * y == x * yfunc
    assert y * x == yfunc * x
    assert x ^ y == x ^ yfunc
    assert y ^ x == yfunc ^ x
    assert x & y == x & yfunc
    assert y & x == yfunc & x
    assert x | y == x | yfunc
    assert y | x == yfunc | x
    assert x @ y == x @ yfunc
    assert y @ x == yfunc @ x
    assert x >> y == x >> yfunc
    assert y >> x == yfunc >> x

def test_43():
    alg = Algebra(2)
    x = alg.vector(name='x')
    assert x.inv() == 1 / x

def test_blades_of_grade():
    alg = Algebra(4)
    grade_combinations = [comb for r in range(alg.d + 2) for comb in itertools.combinations(tuple(range(alg.d + 1)), r=r)]
    for comb in grade_combinations:
        indices = tuple(alg.indices_for_grades(comb))
        blades_of_grade = alg.blades.grade(*indices)
        assert isinstance(blades_of_grade, dict)
        assert all(label in alg.canon2bin and blade.grades[0] in indices
                   for label, blade in blades_of_grade.items())

def test_map_filter():
    alg = Algebra(4)
    x = alg.vector([0, 1, 2, 0])
    xnonzero = x.filter(lambda v: v)
    xoddidx = x.filter(lambda k, v: not k % 2)
    assert xnonzero.values() == [1, 2]
    assert xnonzero.keys() == (2, 4)
    assert xoddidx.values() == [1, 2, 0]
    assert xoddidx.keys() == (2, 4, 8)

    xmul = x.map(lambda v: 2*v)
    coeffs = {1: 2, 2: 2, 4: 2, 8: 4}
    xcoeffmul = x.map(lambda k, v: coeffs[k] * v)
    assert xmul.values() == [0, 2, 4, 0]
    assert xcoeffmul.values() == [0, 2, 4, 0]

def test_simple_exp():
    alg = Algebra(2, 0, 1)
    B = alg.bivector([1, 2, 0])
    T = B.exp()
    assert T == 1 + B

    B = alg.bivector([0, 0, 2])
    R = B.exp()
    assert R == np.cos(2) + np.sin(2) * B.normalized()

    B = alg.bivector([0, 1, 2])
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    assert R == np.cos(l) + np.sin(l) * B / l

    B = alg.bivector([0, 1 + 2j, 2 - 0.5j])
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    Rexpected = np.cos(l) + np.sin(l) * B / l
    assert (R - Rexpected).filter(lambda v: np.abs(v) > 1e-15) == alg.multivector()

    B = alg.bivector(name='B')
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    assert R == cos(l) + sinc(l) * B

def test_dual_numbers():
    alg = Algebra(r=1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    assert x + y == alg.multivector(e=x.e + y.e, e0=x.e0 + y.e0)
    assert x * y == alg.multivector(e=x.e * y.e, e0=x.e * y.e0 + x.e0 * y.e)
    assert x / y == alg.multivector(e=x.e / y.e, e0=(x.e0 * y.e - x.e * y.e0) / y.e**2)
    assert x.sqrt() == alg.multivector(e=x.e**0.5, e0=0.5 * x.e0 / x.e**0.5)
    assert x**3 == alg.multivector(e=x.e ** 3, e0=3 * x.e0 * x.e**2)

def test_power():
    alg = Algebra(2)
    x = alg.multivector(name='x')
    xinv = x.inv()
    assert x ** 3 == x * x * x
    assert x ** 0.5 == x.sqrt()
    assert x ** -0.5 == xinv.sqrt()
    assert x ** -3 == xinv * xinv * xinv

def test_nested_algebra_print():
    dalg = Algebra(0, 0, 1)
    dalg2 = Algebra(0, 0, 1, pretty_blade='𝒇')
    x = dalg.multivector(e='x', e0=1)
    x = dalg2.multivector(e=x, e0=1)
    assert str(x ** 2) == "((x**2) + (2*x) 𝐞₀) + ((2*x) + 2 𝐞₀) 𝒇₀"

def test_free_symbols():
    alg = Algebra(3)
    X = alg.multivector()
    assert X.free_symbols == set()

def test_swap_blades():
    """
    Test the _swap_blades function, which should place like-labels together
    to find the number of swaps and the resulting blade.
    """
    from kingdon.algebra import _swap_blades

    tests = [
        {'input': ('1', '2', '12'), 'output': (0, '12', '')},
        {'input': ('1', '2', '21'), 'output': (1, '21', '')},
        {'input': ('123', '1', '23'), 'output': (2, '23', '1')},
        {'input': ('123', '1', '32'), 'output': (3, '32', '1')},

        {'input': ('23', '1', '123'), 'output': (2, '123', '')},

        {'input': ('', ''), 'output': (0, '', '')},
        {'input': ('', '2'), 'output': (0, '2', '')},
        {'input': ('2', ''), 'output': (0, '2', '')},
        {'input': ('21', '3'), 'output': (0, '213', '')},
        {'input': ('21', '1'), 'output': (0, '2', '1')},
        {'input': ('12', '1'), 'output': (1, '2', '1')},
        {'input': ('1', '21'), 'output': (1, '2', '1')},
        {'input': ('1', '12'), 'output': (0, '2', '1')},
        {'input': ('321', '3'), 'output': (2, '21', '3')},
        {'input': ('231', '3'), 'output': (1, '21', '3')},
        {'input': ('213', '3'), 'output': (0, '21', '3')},
        {'input': ('3', '321'), 'output': (0, '21', '3')},
        {'input': ('3', '231'), 'output': (1, '21', '3')},
        {'input': ('3', '213'), 'output': (2, '21', '3')},
        {'input': ('31', '321'), 'output': (2, '2', '31')},
        {'input': ('321', '31'), 'output': (2, '2', '31')},
        {'input': ('123', '12'), 'output': (3, '3', '12')},
    ]
    for test in tests:
        swaps, res_blade, eliminated = _swap_blades(*test['input'])
        assert swaps == test['output'][0]
        assert res_blade == test['output'][1]
        assert eliminated == test['output'][2]

def test_custom_basis():
    basis = ["e","e1","e2","e3","e0","e01","e02","e03","e12","e31","e23","e032","e013","e021","e123","e0123"]
    pga3d = Algebra.fromname('3DPGA')
    assert pga3d.basis == basis
    assert list(pga3d.canon2bin.keys()) == basis

    e20, e0, e2 = pga3d.blades.e20, pga3d.blades.e0, pga3d.blades.e2
    assert e20 * e2 == - e0

    X = pga3d.multivector(e12=1)
    assert X == pga3d.blades.e12
    assert X == - pga3d.blades.e21
    assert X.e21 == -1
    assert X.e12 == 1

    # Compare a mv in alg301 with pga3d and test if they are identical when we extract the coefficients.
    alg301 = Algebra(3, 0, 1)
    x = alg301.multivector(name='x')
    X = pga3d.multivector(**{alg301.bin2canon[k]: v for k, v in x.items()})
    assert all(getattr(x, blade) == getattr(X, blade) for blade in alg301.canon2bin)
    assert all(getattr(x, blade) == getattr(X, blade) for blade in pga3d.canon2bin)

    # Same, but now after performing a product.
    y = alg301.multivector(name='y')
    Y = pga3d.multivector(**{alg301.bin2canon[k]: v for k, v in y.items()})
    xy, XY = x*y, X*Y
    assert all(getattr(xy, blade) == getattr(XY, blade) for blade in alg301.canon2bin)
    assert all(getattr(xy, blade) == getattr(XY, blade) for blade in pga3d.canon2bin)

def test_apply_to_list():
    alg = Algebra(2, 0, 1)
    line1 = alg.vector(e1=-1, e2=1)  # the line "y - x = 0"
    line2 = alg.vector(e1=-1, e2=2)
    R = (line1 * line2).normalized()
    triangle = [
        alg.vector(e0=1, e1=1).dual(),
        alg.vector(e0=1, e1=0.6, e2=0.5).dual(),
        alg.vector(e0=1, e1=1.3, e2=0.8).dual()
    ]
    operators = [operator.mul, operator.xor, operator.and_, operator.rshift]
    for op in operators:
        transformed_subjects = op(R, triangle)
        for i, p in enumerate(triangle):
            assert op(R, p) == transformed_subjects[i]

        transformed_subjects = op(triangle, R)
        for i, p in enumerate(triangle):
            assert op(p, R) == transformed_subjects[i]


def test_shallow_copy_multivector(pga2d):
    mv = pga2d.vector(e1=[2, 1], e2=[4, 0])
    copied_mv = copy.copy(mv)
    assert mv is not copied_mv
    assert mv.keys() is copied_mv.keys()
    assert mv.values() is copied_mv.values()


def test_deep_copy_multivector(pga2d):
    mv = pga2d.vector(e1=[[2], [1]], e2=[4, 0])
    copied_mv = copy.deepcopy(mv)

    assert mv is not copied_mv
    assert mv == copied_mv

    assert mv.keys() == copied_mv.keys()

    assert mv.values() is not copied_mv.values()
    assert mv.values() == copied_mv.values()

    mv.e1.append(1)
    assert copied_mv.e1[1] == [1]

def test_101():
    # No news is good news, because with kingdon <= 1.5.1 this raised a MemoryError
    alg = Algebra(16)
    assert alg.large
    basisvectors = [alg.vector(keys=(2 ** i,), values=(1.0,)) for i in range(alg.d)]
    e0, e1, e2, e3, *rest = basisvectors
    e01, e02, e03 = e0 ^ e1, e0 ^ e2, e0 ^ e3
    assert e01 == e0 * e1
    spine = sum(basisvectors[2*i] * basisvectors[2*i+1] for i in range(alg.d // 2))
    box = (1j*spine).outerexp().map(lambda v: v.real).filter()
    assert e01 * e02 * box == - e03 * box

def test_large():
    # Force large algebra mode, and compare the results
    d = 2
    alg_large = Algebra(d, large=True)
    alg_small = Algebra(d, large=False)
    x = alg_large.multivector(name='x')
    y = alg_large.multivector(name='y')

    for op_name, op_large in alg_large.registry.items():
        op_small = alg_small.registry.get(op_name)
        if op_name == 'sqrt':
            continue  # Sqrt is a bit of a Heisenbug due to numerical errors

        if isinstance(op_small, UnaryOperatorDict):
            xy_large = op_large(x)
            xy_small = op_small(x)
        else:
            xy_large = op_large(x, y)
            xy_small = op_small(x, y)
        assert not xy_large - xy_small

def test_operators_api():
    d = 2
    alg_large = Algebra(d, large=True)
    alg_small = Algebra(d, large=False)

    # Check consistency of API for both small and large algebras
    for alg in [alg_large, alg_small]:
        x = alg.multivector(name='x')
        y = alg.multivector(name='y')
        # Binary operators should across lists and tuples
        xy = [x, y]
        assert alg.scalar(e=0.5) * xy == [alg.scalar(e=0.5) * z for z in xy]
        # Unary operators can be called on lists as well
        assert alg.normsq(xy) == [z.normsq() for z in xy]
        # Multiplying a function with a MV is allowed
        func = lambda: alg.scalar(e=0.5)
        assert x * func == x * alg.scalar(e=0.5)
        assert alg.normsq(func) == alg.normsq(alg.scalar(e=0.5))

def test_87(pga2d):
    # Test if MVs operators are used when one of the arguments is a numpy ndarray.
    len_vector = len(pga2d.blades.grade(1))
    uvals = np.random.random((len_vector, 2))
    u = pga2d.vector(uvals)
    one = np.ones(2)
    expectedmv = pga2d.scalar(e=one) + pga2d.vector(uvals)
    diff1 = (u + one) - expectedmv
    diff2 = (one + u) - expectedmv
    assert all(diff1.map(lambda v: np.allclose(v, 0.0)).values())
    assert all(diff2.map(lambda v: np.allclose(v, 0.0)).values())
