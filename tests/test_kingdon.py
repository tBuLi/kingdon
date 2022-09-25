#!/usr/bin/env python

"""Tests for `kingdon` package."""
from dataclasses import replace

import pytest
import numpy as np

from sympy import Symbol, simplify, factor, expand, collect
from kingdon import Algebra, MultiVector

import timeit

@pytest.fixture
def pga1d():
    return Algebra(1, 0, 1)


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

def test_algebra_constructors(pga1d):
    assert pga1d.scalar.keywords['grade'] == 0
    assert pga1d.vector.keywords['grade'] == 1
    assert pga1d.bivector.keywords['grade'] == 2
    assert pga1d.pseudoscalar.keywords['grade'] == 2
    assert pga1d.pseudovector.keywords['grade'] == 1
    assert pga1d.pseudobivector.keywords['grade'] == 0

    assert not hasattr(pga1d, 'trivector')
    assert not hasattr(pga1d, 'pseudotrivector')

    x = pga1d.vector([1, 2])
    assert len(x) == 2
    assert isinstance(x, MultiVector)

def test_algebra_symbolic():
    alg = Algebra(3, 0, 1)


def test_MultiVector(pga1d):
    X = MultiVector(algebra=pga1d)
    assert isinstance(X.vals, dict)
    assert X.name == ''
    assert len(X) == len(X.vals)
    with pytest.raises(TypeError):
        # If starting from a sequence, it must be of length len(algebra)
        X = MultiVector(algebra=pga1d, vals=[1, 2])
    with pytest.raises(TypeError):
        # vals must be iterable (sequence)
        X = MultiVector(algebra=pga1d, vals=2)
    with pytest.raises(TypeError):
        # No algebra provided
        X = MultiVector(name='X')
    with pytest.raises(KeyError):
        # Dict keys must be either (binary) numbers or canonical basis element strings.
        X = MultiVector(vals={'a': 2, 'e12': 1}, algebra=pga1d)
    X = MultiVector(vals={0: 2.2, 'e12': 1.2}, algebra=pga1d)
    assert X.vals == {0: 2.2, 3: 1.2}

def test_anticommutation(pga1d, vga11, vga2d):
    for alg in [pga1d, vga11, vga2d]:
        X = alg.multivector({1: 1})
        Y = alg.multivector({2: 1})
        assert X*Y == -Y*X

def test_gp(pga1d):
    # Multiply two multivectors
    X = MultiVector(vals={'1': 2, 'e12': 3}, algebra=pga1d)
    Y = MultiVector(vals={'1': 7, 'e12': 5}, algebra=pga1d)
    Z = X.gp(Y)
    assert Z.vals == {0: 2*7, 3: 2*5 + 3*7}

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
    Z3 = R >> X
    for i, xrow in enumerate(valsX.T):
        Rx = R >> vga2d.vector(xrow)
        assert Rx[1] == Z3[1][i]

def test_reverse(R6):
    X = R6.multivector(np.arange(0, 2 ** 6))
    Xrev = ~X
    assert X.grade((0, 1, 4, 5)) == Xrev.grade((0, 1, 4, 5))
    assert X.grade((2, 3, 6)) == - Xrev.grade((2, 3, 6))

def test_indexing(pga1d):
    # Test indexing of a mv with canonical and binary indices.
    X = pga1d.multivector()
    X[0], X['e12'] = 2, 3
    assert X['1'] == 2 and X[3] == 3

def test_gp_symbolic(vga2d):
    u = vga2d.vector(name='u')
    u1, u2 = u.vals[1], u.vals[2]
    usq = u*u
    # Square of a vector should be purely scalar.
    assert usq[0] == u1**2 + u2**2
    assert len(usq) == 1
    with pytest.raises(KeyError):
        usq['e12']

    # A bireflection should have both a scalar and bivector part however.
    v = vga2d.vector(name='v')
    v1, v2 = v.vals[1], v.vals[2]
    R = u*v
    assert R[0] == u1 * v1 + u2 * v2
    assert R[3] == u1 * v2 - u2 * v1

    # The norm of a bireflection is a scalar.
    Rnormsq = R*~R
    assert Rnormsq[0] == (u1*v1 + u2*v2)**2 - (-u1*v2 + u2*v1)*(u1*v2 - u2*v1)
    with pytest.raises(KeyError):
        Rnormsq['e12']

def test_sp_symbolic(vga2d):
    u = vga2d.vector(name='u')
    v = vga2d.vector(name='v')
    # Pure vector
    assert (u >> v) == (u >> v).grade(1)

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
    assert e1^e2 == -e2^e1

    # Test basis bivectors.
    e12, e23 = sta.blades['e12'], sta.blades['e23']
    assert (e12 ^ e23).vals == {}  # TODO: support == 0?
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

def test_fromtrusted(vga2d):
    x = vga2d.mvfromtrusted(vals={1: 1.1})
    print(x)
