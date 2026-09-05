#!/usr/bin/env python

"""Tests for `kingdon` package."""
import copy
import itertools
import operator
from dataclasses import replace

import pytest
import numpy as np

from sympy import Symbol, expand, sympify, cos, sin, sinc
from kingdon import (Algebra, MultiVector, Scalar, Vector, Bivector, Trivector, Quadvector,
                     Bireflection, Direction, EVector, UPoint, Point, Translation, symbols)
from kingdon.operator_dict import UnaryOperatorDict
from kingdon.numerical import exp as numerical_exp, is_close

import timeit

@pytest.fixture
def ga101():
    return Algebra(signature=[1, 0], start_index=1)


@pytest.fixture
def ga201():
    return Algebra(signature=[1, 1, 0], start_index=1)


@pytest.fixture
def ga301():
    return Algebra(signature=[1, 1, 1, 0], start_index=1)


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


def test_MultiVector(ga101):
    with pytest.raises(TypeError):
        # If starting from a sequence, it must be of length len(algebra)
        X = MultiVector(algebra=ga101, values=[1, 2])
    with pytest.raises(TypeError):
        # vals must be iterable (sequence)
        X = MultiVector(algebra=ga101, values=2)
    with pytest.raises(TypeError):
        # No algebra provided
        X = MultiVector(name='X')
    with pytest.raises(KeyError):
        # Dict keys must be either (binary) numbers or canonical basis element strings.
        X = MultiVector(values={'a': 2, 'e12': 1}, algebra=ga101)
    X = MultiVector(values={0: 2.2, 'e12': 1.2}, algebra=ga101)
    assert dict(X.items()) == {0: 2.2, 3: 1.2}

def test_anticommutation(ga101, vga11, vga2d):
    for alg in [ga101, vga11, vga2d]:
        X = alg.multivector({1: 1})
        Y = alg.multivector({2: 1})
        assert X*Y == -Y*X

def test_gp(ga101):
    # Multiply two multivectors. Also tests two different MV creation API's
    X = MultiVector(values={'e': 2, 'e12': 3}, algebra=ga101)
    Y = MultiVector(e=7, e12=5, algebra=ga101)
    Z = X * Y
    assert dict(Z.items()) == {0: 2*7, 3: 2*5 + 3*7}

def test_cayley(ga101, vga2d, vga11):
    assert ga101.cayley == {('e', 'e'): 'e', ('e', 'e1'): 'e1', ('e', 'e12'): 'e12', ('e', 'e2'): 'e2',
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

def test_purevector(ga101):
    with pytest.raises(TypeError):
        # Grade needs to be specified.
        ga101.purevector({1: 1, 2: 1})
    with pytest.raises(ValueError):
        # Grade must be valid, in this case no larger than 2.
        ga101.purevector({1: 1, 2: 1}, grade=10)
    with pytest.raises(TypeError):
        # vals must be of the specified grade.
        ga101.purevector({0: 1, 2: 1}, grade=1)
    x = ga101.purevector({1: 1, 2: 1}, grade=1)
    assert type(x) == Vector
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

def test_getattr_setattr(ga101):
    X = ga101.multivector({0: 2, 'e12': 3})
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
    assert len(usq.values()) == 1
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
    assert len(Rnormsq.values()) == 1
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
    assert vga2d.blades['e'] == vga2d.scalar({'e': 1})
    assert vga2d.blades['e1'] == vga2d.vector({'e1': 1})
    assert vga2d.blades['e2'] == vga2d.vector({'e2': 1})
    assert vga2d.blades['e12'] == vga2d.bivector({'e12': 1})
    assert vga2d.blades['e12'] == vga2d.pss

def _assert_blade_typing(alg, basis_blade, expected, expected_full):
    """ Assert that :code:`alg.blades[basis_blade]` has the expected value *and* type. """
    blade = alg.blades[basis_blade]
    expectation = (expected_full if alg.full_layout else expected)(alg)
    assert type(blade) is type(expectation)
    assert blade == expectation


@pytest.mark.parametrize("alg", [Algebra.fromname("2DPGA"), Algebra.fromname("2DPGA", full_layout=True)])
@pytest.mark.parametrize("basis_blade, expected, expected_full", [
    ('e', lambda alg: Translation(alg), lambda alg: Translation(alg, [0, 0])),
    ('e0', lambda alg: UPoint(alg), lambda alg: UPoint(alg, [0, 0])),
    ('e1', lambda alg: EVector(alg, e1=1), lambda alg: EVector(alg, [1, 0])),
    ('e2', lambda alg: EVector(alg, e2=1), lambda alg: EVector(alg, [0, 1])),
    ('e12', lambda alg: Point(alg), lambda alg: Point(alg, [0, 0])),
    ('e01', lambda alg: Direction(alg, e01=1), lambda alg: Direction(alg, [0, 1])),
    ('e02', lambda alg: Direction(alg, e02=1), lambda alg: Direction(alg, [-1, 0])),
    ('e012', lambda alg: Trivector(alg, e012=1), lambda alg: Trivector(alg, e012=1)),
    ('e012', lambda alg: Trivector(alg, e012=1), lambda alg: alg.pss),
    # Blades whose indices are not in canonical order pick up a sign. Such a blade
    # is not always of the same type as its canonical counterpart, since the
    # normalized types (Point, Translation, ...) fix a coefficient to +1 and hence
    # negating them falls back to the corresponding k-vector.
    ('e10', lambda alg: Direction(alg, e10=1), lambda alg: Direction(alg, [0, -1])),
    ('e20', lambda alg: Direction(alg, e20=1), lambda alg: Direction(alg, [1, 0])),
    ('e21', lambda alg: Bivector(alg, e21=1), lambda alg: Bivector(alg, [0, 0, -1])),
    ('e021', lambda alg: Trivector(alg, e021=1), lambda alg: Trivector(alg, e012=-1)),
    ('e120', lambda alg: Trivector(alg, e012=1), lambda alg: Trivector(alg, e012=1)),
])
def test_blades_typing_2dpga_named(alg, basis_blade, expected, expected_full):
    """ Ensure that the basis blades are correctly typed. """
    _assert_blade_typing(alg, basis_blade, expected, expected_full)


@pytest.mark.parametrize("alg", [Algebra(2,0,1), Algebra(2,0,1, full_layout=True)])
@pytest.mark.parametrize("basis_blade, expected, expected_full", [
    ('e', lambda alg: Scalar(alg, e=1), lambda alg: Scalar(alg, [1])),
    ('e0', lambda alg: Vector(alg, e0=1), lambda alg: Vector(alg, [1, 0, 0])),
    ('e1', lambda alg: Vector(alg, e1=1), lambda alg: Vector(alg, [0, 1, 0])),
    ('e2', lambda alg: Vector(alg, e2=1), lambda alg: Vector(alg, [0, 0, 1])),
    ('e12', lambda alg: Bivector(alg, e12=1), lambda alg: Bivector(alg, [0, 0, 1])),
    ('e01', lambda alg: Bivector(alg, e01=1), lambda alg: Bivector(alg, [1, 0, 0])),
    ('e02', lambda alg: Bivector(alg, e02=1), lambda alg: Bivector(alg, [0, 1, 0])),
    ('e012', lambda alg: Trivector(alg, e012=1), lambda alg: Trivector(alg, e012=1)),
    ('e012', lambda alg: Trivector(alg, e012=1), lambda alg: alg.pss),
])
def test_blades_typing_2dpga(alg, basis_blade, expected, expected_full):
    """ Ensure that the basis blades are correctly typed. """
    _assert_blade_typing(alg, basis_blade, expected, expected_full)


# In 3DPGA the bivectors are just bivectors, whereas in 2DPGA they are directions.
@pytest.mark.parametrize("alg", [Algebra.fromname('3DPGA'), Algebra.fromname('3DPGA', full_layout=True)])
@pytest.mark.parametrize("basis_blade, expected, expected_full", [
    ('e', lambda alg: Translation(alg), lambda alg: Translation(alg, [0, 0, 0])),
    ('e0', lambda alg: UPoint(alg), lambda alg: UPoint(alg, [0, 0, 0])),
    ('e1', lambda alg: EVector(alg, e1=1), lambda alg: EVector(alg, [1, 0, 0])),
    ('e2', lambda alg: EVector(alg, e2=1), lambda alg: EVector(alg, [0, 1, 0])),
    ('e3', lambda alg: EVector(alg, e3=1), lambda alg: EVector(alg, [0, 0, 1])),
    ('e01', lambda alg: Bivector(alg, e01=1), lambda alg: Bivector(alg, [1, 0, 0, 0, 0, 0])),
    ('e02', lambda alg: Bivector(alg, e02=1), lambda alg: Bivector(alg, [0, 1, 0, 0, 0, 0])),
    ('e03', lambda alg: Bivector(alg, e03=1), lambda alg: Bivector(alg, [0, 0, 1, 0, 0, 0])),
    ('e12', lambda alg: Bivector(alg, e12=1), lambda alg: Bivector(alg, [0, 0, 0, 1, 0, 0])),
    ('e31', lambda alg: Bivector(alg, e31=1), lambda alg: Bivector(alg, [0, 0, 0, 0, 1, 0])),
    ('e23', lambda alg: Bivector(alg, e23=1), lambda alg: Bivector(alg, [0, 0, 0, 0, 0, 1])),
    ('e021', lambda alg: Direction(alg, e021=1), lambda alg: Direction(alg, [0, 0, 1])),
    ('e013', lambda alg: Direction(alg, e013=1), lambda alg: Direction(alg, [0, 1, 0])),
    ('e032', lambda alg: Direction(alg, e032=1), lambda alg: Direction(alg, [1, 0, 0])),
    ('e123', lambda alg: Point(alg), lambda alg: Point(alg, [0, 0, 0])),
    ('e0123', lambda alg: Quadvector(alg, e0123=1), lambda alg: Quadvector(alg, e0123=1)),
    ('e0123', lambda alg: Quadvector(alg, e0123=1), lambda alg: alg.pss),
    # Non-canonical basis blades, see the 2DPGA case above.
    ('e10', lambda alg: Bivector(alg, e10=1), lambda alg: Bivector(alg, [-1, 0, 0, 0, 0, 0])),
    ('e21', lambda alg: Bivector(alg, e21=1), lambda alg: Bivector(alg, [0, 0, 0, -1, 0, 0])),
    ('e30', lambda alg: Bivector(alg, e30=1), lambda alg: Bivector(alg, [0, 0, -1, 0, 0, 0])),
    ('e012', lambda alg: Direction(alg, e012=1), lambda alg: Direction(alg, [0, 0, -1])),
    ('e321', lambda alg: Trivector(alg, e321=1), lambda alg: Trivector(alg, [0, 0, 0, -1])),
    ('e1023', lambda alg: Quadvector(alg, e1023=1), lambda alg: Quadvector(alg, e0123=-1)),
])
def test_blades_typing_3dpga(alg, basis_blade, expected, expected_full):
    """ Ensure that the basis blades are correctly typed. """
    _assert_blade_typing(alg, basis_blade, expected, expected_full)


def _assert_canonical(mv, label=''):
    canon_pos = {k: i for i, k in enumerate(mv.algebra.canon2bin.values())}
    positions = [canon_pos[k] for k in mv.keys()]
    assert positions == sorted(positions), f'{label}: keys {mv.keys()} not in canon2bin order (positions={positions})'


@pytest.mark.parametrize("alg", [
    Algebra(2, 0, 1),
    Algebra(3, 0, 1),
    Algebra(3),
    Algebra(2, 1),
    Algebra.fromname('2DPGA'),
    Algebra.fromname('3DPGA'),
])
def test_canonical_key_order(alg):
    """
    However a multivector was made, its keys should be listed in the order in
    which they appear in canon2bin.

    That order is not the same as simply sorting the binary keys: e.g. the bivectors
    of 3DPGA are 3, 5, 9, 6, 10, 12, so anything that sorts numerically, or that
    merges two sets of keys without care, quietly ends up pairing the values with
    the wrong basis blades. And since equality compares the key tuples as they
    are, such a multivector no longer compares equal to its well ordered twin
    either.

    So the usual suspects are all checked here: the binary and unary operators,
    grade projection, and every basis blade in the blade dict. The add and sub at
    the end get their own line because their arguments have no grades in common,
    and interleaving two different sets of keys is exactly where the order tends
    to slip.
    """
    from kingdon import operators as ops

    x = alg.multivector(name='x')
    y = alg.multivector(name='y')

    binary_codegens = [
        ops.gp, ops.cp, ops.acp, ops.ip,
        ops.sp, ops.lc, ops.rc, ops.op,
        ops.rp, ops.add, ops.sub,
    ]
    for op in binary_codegens:
        _assert_canonical(op(x, y), label=op.__name__)

    unary_codegens = [
        ops.neg, ops.reverse, ops.involute,
        ops.conjugate, ops.hodge, ops.unhodge,
        ops.polarity, ops.unpolarity, ops.normsq,
        ops.sqrt,
    ]
    for op in unary_codegens:
        _assert_canonical(op(x), label=op.__name__)

    for grade in range(alg.d + 1):
        _assert_canonical(x.grade(grade), label=f'grade({grade})')

    for blade in alg.canon2bin:
        _assert_canonical(alg.blades[blade], label=f'blades[{blade}]')

    _assert_canonical(ops.add(x.grade(0), x.grade(2)), label='add(g0,g2)')
    _assert_canonical(ops.sub(x.grade(1), x.grade(2)), label='sub(g1,g2)')


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

def test_alg_full_layout(vga2d):
    vga2d_full = replace(vga2d, full_layout=True)
    assert vga2d != vga2d_full
    u = vga2d_full.vector([1, 2])
    v = vga2d_full.vector([0, 3])
    R = u * v
    assert R.grades == (0, 2)
    assert R.e == 6
    assert R.e1 == 0
    assert R.e2 == 0
    assert R.e12 == 3


def test_alg_graded_deprecated():
    with pytest.warns(FutureWarning):
        alg = Algebra(2, graded=True)
    assert alg.full_layout
    with pytest.warns(FutureWarning):
        alg = Algebra.fromname('2DPGA', graded=False)
    assert not alg.full_layout


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

def test_hodge_dual(ga201, ga301):
    x = ga201.multivector(name='x')
    y = x.dual()
    # GAmphetamine.js output
    assert dict(y.items()) == {0: x.e123, 1: x.e23, 2: -x.e13, 4: x.e12, 3: x.e3, 5: -x.e2, 6: x.e1, 7: x.e}
    z = y.undual()
    assert x == z
    with pytest.raises(ValueError):
        x.dual('poincare')

    # Test hodge dual in 3DPGA
    x = ga301.multivector(name='x')
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

def test_regressive(ga301):
    """ Test the regressive product of full mvs in 3DPGA against the known result from GAmphetamine.js"""
    xvals = symbols(','.join(f'x{i}' for i in range(1, len(ga301) + 1)))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = xvals
    x = ga301.multivector({k: xvals[i] for i, k in enumerate(ga301.canon2bin)})
    yvals = symbols(','.join(f'y{i}' for i in range(1, len(ga301) + 1)))
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16 = yvals
    y = ga301.multivector({k: yvals[i] for i, k in enumerate(ga301.canon2bin)})

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
    known = ga301.multivector(known_vals)
    x_regr_y = x & y
    assert x_regr_y == known


def test_projection_3d(ga301):
    x1, x2, x3 = symbols('x1, x2, x3')
    x = ga301.trivector([x1, x2, x3, 1])
    y1, y2, y3, y4 = symbols('y1, y2, y3, y4')
    y = ga301.vector([y1, y2, y3, y4])
    # project the plane y onto the point x
    z = y @ x
    assert z.grades == (1,)
    # project the point x onto the plane y
    z = x @ y
    assert z.grades == (3,)


def test_inv_div(ga201):
    u = ga201.multivector(name='u')
    # Multiply by inverse results in a scalar exp, which numerically evaluates to 1.
    def uinv(u): return u*u.inv()
    func_uinv = ga201.compile(uinv, u)
    u_num = ga201.multivector(np.random.random(8))
    res = func_uinv(u_num)
    assert isinstance(res, Scalar)
    assert res.e == pytest.approx(1.0)
    assert res.grades == (0,)
    # Division by self is truly the scalar 1.
    def udivu(u): return u / u
    func_udivu = ga201.compile(udivu, u)
    res = func_udivu(u_num)
    assert isinstance(res, Scalar)
    assert res.e == pytest.approx(1.0)
    assert res.grades == (0,)

def test_hitzer_inv():
    from kingdon.polynomial import RationalPolynomial
    for d in range(5): # The d=5 case is excluded because the test is too slow.
        alg = Algebra(d)
        x = alg.multivector(name='x', symbolcls=RationalPolynomial.fromname)
        assert not (x * x.inv() - alg.blades.e)


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

    assert x._values == list(xvals)
    assert x._keys is xkeys

    # We use sympify, so string that look like equations are also allowed
    y = alg.multivector(['a*b+c', '-15*c'], grades=(1,))
    assert y.e1 == Symbol('a')*Symbol('b') + Symbol('c')
    assert y.e2 == -15 * Symbol('c')

    yvals = symbols('y y1 y2 y12')
    with pytest.raises(TypeError):
        y = alg.multivector(yvals, xkeys[:3])
    y = alg.multivector(yvals)
    assert y._values == list(yvals)
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
    alg = Algebra(2, 0, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')
    with pytest.raises(TypeError):
        x >> y

    x = alg.evenmv(name='x')
    # GAmphetamine sw formula for even versors: grade(x*y*~x + y*(1 - grade(x*~x, 0)), grade(y))
    # For normalized x (x*~x = 1 scalar), the correction term vanishes and this reduces to x*y*~x.
    axar_scalar = (x * ~x).grade(0)
    xconjy_expected = sum((x * y.grade(g) * ~x + y.grade(g) * (1 - axar_scalar)).grade(g) for g in y.grades)
    xconjy = x >> y
    diff = (xconjy_expected - xconjy)
    assert not diff

    x = alg.oddmv(name='x')
    axar_scalar = (x * ~x).grade(0)
    xconjy_expected = ((x * y.grade(0) * ~x + y.grade(0) * (1 - axar_scalar))
                       - (x * y.grade(1) * ~x + y.grade(1) * (1 - axar_scalar))
                       + (x * y.grade(2) * ~x + y.grade(2) * (1 - axar_scalar))
                       - (x * y.grade(3) * ~x + y.grade(3) * (1 - axar_scalar)))
    xconjy = x >> y
    diff = (xconjy_expected - xconjy).map(lambda v: v.simplify())
    assert not diff


def test_projection():
    alg = Algebra(2, 0, 1)
    x = alg.multivector(name='x')  # multivector
    y = alg.multivector(name='y')
    with pytest.raises(TypeError):
        x @ y

    y = alg.evenmv(name='y')
    ynormsq = (y * ~y).grade(0)
    xprojy_expected = (x | y) * ~y + x*(1 - ynormsq)
    xprojy = x @ y
    diff = (xprojy_expected - xprojy)
    assert not diff

    y = alg.oddmv(name='y')
    ynormsq = (y * ~y).grade(0)
    xprojy_expected = (x | y) * ~y + x*(1 - ynormsq)
    xprojy = x @ y
    diff = (xprojy_expected - xprojy)
    assert not diff


def test_outerexp(R6):
    from sympy import Number
    B = R6.bivector(name='B')
    LB = B.outerexp()
    LB_exact = 1 + B + (B ^ B) / Number(2) + (B ^ B ^ B) / Number(6)

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


def test_normalization(ga301):
    vvals = np.random.random(len(tuple(ga301.indices_for_grade(1))))
    v = ga301.vector(vvals).normalized()
    assert (v*v).e == pytest.approx(1.0)

    # Normalizing a non-simple bivector makes it simple!
    bvals = np.random.random(len(tuple(ga301.indices_for_grade(2))))
    B = ga301.bivector(bvals)
    Bnormalized = B.normalized()
    assert Bnormalized.normsq().e == pytest.approx(1.0)
    assert Bnormalized.normsq().e1234 == pytest.approx(0.0)


def test_iteration():
    alg = Algebra(4)
    nrows = 3
    shape = (len(tuple(alg.indices_for_grade(2))), nrows)
    bvals = np.random.random(shape)
    B = alg.bivector(bvals)
    for i, b in enumerate(B):
        np.testing.assert_allclose(b.values(), bvals[:, i])
    assert i + 1 == nrows


def test_fromsignature():
    alg = Algebra(signature=[0, -1, 1, 1])
    assert alg.start_index == 0
    assert isinstance(alg.signature, list)
    assert np.all(alg.signature == [0, -1, 1, 1])
    assert (alg.p, alg.q, alg.r) == (2, 1, 1)
    with pytest.raises(TypeError):
        alg = Algebra(signature=[0, -1, 1, 1, 2])


def test_start_index():
    pga2d = Algebra(signature=[0, 1, 1], start_index=0)
    alg = Algebra(signature=[0, 1, 1], start_index=1)
    for ei, fi in zip(pga2d.blades.values(), alg.blades.values()):
        assert (fi**2).e == (ei**2).e


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


@pytest.mark.parametrize("alg", [
    Algebra(2, 0, 1, full_layout=True),
    Algebra(3, full_layout=True),
    Algebra.fromname("2DPGA", full_layout=True),
])
def test_full_layout(alg):
    for b in alg.blades.values():
        assert b.keys() == tuple(k for k, v in alg._type_layouts[type(b)].items() if v == ...)

    with pytest.raises(ValueError):
        # In full_layout mode, the keys have to be correct.
        x = alg.multivector(name='x', keys=(1,))


def test_blade_dict():
    alg = Algebra(2)
    assert not alg.blades.lazy
    assert len(alg.blades) == len(alg)
    locals().update(**alg.blades)
    with pytest.raises(AttributeError):
        alg.blades.f123

    alg = Algebra(2, full_layout=True)
    assert not alg.blades.lazy
    assert len(alg.blades) == len(alg)
    assert len(alg.blades['e1'].values()) == 2

    # In algebras larger than 6, lazy is the default.
    alg = Algebra(7)
    assert alg.blades.lazy
    assert len(alg.blades) == 8  # PSS and basis vectors are calculated by default
    assert len(alg.blades['e12'].values()) == 1
    assert len(alg.blades) == 9

    with pytest.raises(TypeError):
        Algebra(7, full_layout=True)  # A large algebra has no types to lay out.


@pytest.mark.parametrize("symbolic", [True, False])
def test_compile_operator_existence(symbolic):
    """ Test if all battery-included GA operators can be used in custom functions."""
    alg = Algebra(3, 0, 0)
    u = alg.evenmv(name='u')
    v = alg.evenmv(name='v')

    operators = alg.registry.copy()
    for op_name, op_dict in operators.items():
        if op_name == 'sqrt': continue  # Do not agree because one actually does the sqrt while the other postpones.
        if isinstance(op_dict, UnaryOperatorDict):
            def myfunc(x):
                return getattr(x, op_name)()

            myfunc_compiled = alg.add_operator(myfunc, symbolic=symbolic)
            assert not (myfunc_compiled(u).filter(alg.simp_func) - myfunc(u))  # OperatorDict.__call__ does a symbolic simplfication on the result, causing expressions not to be structurely identical. Hence we have to check the difference and not equality.

        else:
            def myfunc(x, y):
                return getattr(x, op_name)(y)
            myfunc_compiled = alg.add_operator(myfunc, symbolic=symbolic)
            assert not myfunc_compiled(u, v) - myfunc(u, v)


@pytest.mark.parametrize("symbolic", [True, False])
def test_compile_basics(symbolic):
    alg = Algebra(3, 0, 1)
    u = alg.multivector(name='u')
    v = alg.multivector(name='v')

    @alg.add_operator(symbolic=symbolic)
    def square(x):
        return x * x

    @alg.add_operator(symbolic=symbolic)
    def double(x):
        return 2 * x

    @alg.add_operator(symbolic=symbolic)
    def add(x, y):
        return x + y

    @alg.add_operator(symbolic=symbolic)
    def grade_select(x):
        return x.grade(1, 2)

    # Test if we can nest compiled expressions.
    @alg.add_operator(symbolic=symbolic)
    def coupled(u, v):
        uv = add(u, v)
        return square(uv) + double(u)

    assert square(u) == square.codegen(u)
    assert double(u) == double.codegen(u)
    assert add(u, v) == add.codegen(u, v)
    assert grade_select(u) == u.grade(1, 2)
    assert not (coupled(u, v) - ((u + v) ** 2 + 2 * u))

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
    empty = alg.scalar()
    assert res == empty
    zero = alg.scalar(e=0)
    assert res != zero


def test_reciprocal_frame():
    alg = Algebra(1, 3)
    for (i, ei), (j, Ej) in itertools.product(enumerate(alg.frame), enumerate(alg.reciprocal_frame)):
        if i == j:
            assert (ei | Ej).e == 1
        else:
            assert (ei | Ej).e == 0

def test_setitem():
    alg = Algebra(2, 0, 1)
    l = 6
    d = 3 / l
    point_vals = np.zeros((alg.d, l + 1))
    point_vals[0] = 1
    point_vals[1] = np.arange(l + 1) * d - 1.5
    # Test with the 2d array first
    lines = alg.vector(point_vals)
    lines[-1] = lines[-2]
    assert lines[-1].keys() == lines[-2].keys()
    assert np.allclose(lines[-1].values(), lines[-2].values())
    lines[:3] = lines[4:]
    assert np.allclose(lines[:3].values(), lines[4:].values())

    # Dualization turns the 2d array into a list of 1d arrays, so we need to test with that as well.
    points = alg.vector(point_vals).dual()
    points[-1] = points[-2]
    assert points[-1].keys() == points[-2].keys()
    assert np.allclose(points[-1].values(), points[-2].values())
    points[:3] = points[4:]
    assert np.allclose(points[:3].values(), points[4:].values())

def test_set_mv():
    alg = Algebra(2, 0, 1)
    x = alg.multivector(name='x')
    y = alg.multivector(name='y')
    x.set(y)
    assert x == y
    assert x._values is not y._values

def test_mv_times_func():
    """If a mv is binaried with a function, we simply call it until it returns a multivector. """
    alg = Algebra(2, 0, 1)  # Smallest non-Abelian algebra, that property is important.
    x = alg.evenmv(name='x')
    y = alg.oddmv(name='x')
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
        blades_of_grade_alt = alg.blades.grade(indices)
        assert blades_of_grade == blades_of_grade_alt
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
    alg = Algebra(2, 0, 1, extra_types=[Translation])
    B = alg.bivector([1, 2, 0]).filter()
    assert isinstance(B, Bivector)
    T = B.exp()
    assert isinstance(T, Translation)
    assert not (T - (1 + B)).filter()

    B = alg.bivector([0, 0, 2]).filter()
    R = B.exp()
    assert isinstance(R, Bireflection)
    assert R == np.cos(2) + np.sin(2) * B.normalized()

    B = alg.bivector([0, 1, 2])
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    assert isinstance(R, Bireflection)
    assert R == np.cos(l) + np.sin(l) * B / l

    B = alg.bivector([0, 1 + 2j, 2 - 0.5j])
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    Rexpected = np.cos(l) + np.sin(l) * B / l
    assert isinstance(Rexpected, Bireflection)
    assert not (R - Rexpected).filter(lambda v: np.abs(v) > 1e-15)

    B = alg.bivector(name='B')
    R = B.exp()
    l = (-(B | B).e) ** 0.5
    assert isinstance(R, Bireflection)
    assert R == cos(l) + sinc(l) * B

def test_numerical_exp_simple_bivector(sta):
    """Test numerical.exp() on a simple bivector matches MultiVector.exp()"""
    B = sta.bivector([1.0, 0, 0, 0, 0, 0])
    mv_exp = B.exp()
    num_exp = numerical_exp(B, n=20)
    assert is_close(mv_exp, num_exp, tol=1e-10)

def test_numerical_exp_non_simple_bivector(sta):
    """Test numerical.exp() on a non-simple bivector doesn't raise an error"""
    B = sta.bivector([1.0, 2.0, 0, 0, 0, 0])
    result = numerical_exp(B, n=20)
    assert result is not None

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
    with pytest.raises(ValueError):
        Algebra.fromname('fantasyalgebra')
    basis_2dpga = ["e", "e1", "e2", "e0", "e20", "e01", "e12", "e012"]
    pga2d = Algebra.fromname('2DPGA')
    alg201 = Algebra(2, 0, 1)
    basis_3dpga = ["e","e1","e2","e3","e0","e01","e02","e03","e12","e31","e23","e032","e013","e021","e123","e0123"]
    pga3d = Algebra.fromname('3DPGA')
    alg301 = Algebra(3, 0, 1)
    basis_stap = [
        "e", "e0", "e1", "e2", "e3", "e4",
        "e01", "e02", "e03", "e40", "e12", "e31", "e23", "e41", "e42", "e43",
        "e234", "e314", "e124", "e123", "e014", "e024", "e034", "e032", "e013", "e021",
        "e0324", "e0134", "e0214", "e0123", "e1234", "e01234"
    ]
    stap = Algebra.fromname('STAP')
    alg311 = Algebra(3, 1, 1)

    for basis, pga, alg in [(basis_2dpga, pga2d, alg201), (basis_3dpga, pga3d, alg301), (basis_stap, stap, alg311)]:
        assert pga.basis == basis
        assert list(pga.canon2bin.keys()) == basis

        e20, e0, e2 = pga.blades.e20, pga.blades.e0, pga.blades.e2
        assert e20 * e2 == - e0

        X = pga.multivector(e21=-1.0)
        assert not (X - pga.blades.e12).filter()
        assert not (X + pga.blades.e21).filter()
        assert X.e21 == -1
        assert X.e12 == 1

        # Compare a mv in alg301 with pga3d and test if they are identical when we extract the coefficients.
        x = alg.multivector(name='x')
        X = pga.multivector(**{alg.bin2canon[k]: v for k, v in x.items()})
        x_dual = x.dual()
        X_dual = X.dual()
        assert x == x_dual.undual()
        assert X == X_dual.undual()
        assert all(getattr(x, blade) == getattr(X, blade) for blade in alg.canon2bin)
        assert all(getattr(x, blade) == getattr(X, blade) for blade in pga.canon2bin)
        assert all(getattr(x_dual, blade) == getattr(X_dual, blade) for blade in alg.canon2bin)
        assert all(getattr(x_dual, blade) == getattr(X_dual, blade) for blade in pga.canon2bin)

        # Same, but now after performing a product.
        for op in [operator.mul, operator.xor, operator.and_, operator.add, operator.sub, operator.or_]:
            y = alg.multivector(name='y')
            Y = pga.multivector(**{alg.bin2canon[k]: v for k, v in y.items()})
            xy, XY = op(x, y), op(X, Y)
            assert all(getattr(xy, blade) == getattr(XY, blade) for blade in alg.canon2bin)
            assert all(getattr(xy, blade) == getattr(XY, blade) for blade in pga.canon2bin)

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


def test_shallow_copy_multivector(ga201):
    mv = ga201.vector(e1=[2, 1], e2=[4, 0])
    copied_mv = copy.copy(mv)
    assert mv is not copied_mv
    assert mv.keys() is copied_mv.keys()
    assert mv.values() is copied_mv.values()


def test_deep_copy_multivector(ga201):
    mv = ga201.vector(e1=[[2], [1]], e2=[4, 0])
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
        if op_name in ['proj', 'sw']:
            continue # These are defined in terms of other operators and thus do not have to be tested separately.

        if isinstance(op_small, UnaryOperatorDict):
            xy_large = op_large(x)
            xy_small = op_small(x)
        else:
            xy_large = op_large(x, y)
            xy_small = op_small(x, y)
        assert isinstance(xy_small, MultiVector)
        assert isinstance(xy_large, MultiVector)
        assert not (xy_large - xy_small).filter()  # filter because direct compute and CSE might result in structurally different expressions that are algebraically the same.

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

def test_87(ga201):
    # Test if MVs operators are used when one of the arguments is a numpy ndarray.
    len_vector = len(ga201.blades.grade(1))
    uvals = np.random.random((len_vector, 2))
    u = ga201.vector(uvals)
    one = np.ones(2)
    expectedmv = ga201.scalar(e=one) + ga201.vector(uvals)
    diff1 = (u + one) - expectedmv
    diff2 = (one + u) - expectedmv
    assert all(diff1.map(lambda v: np.allclose(v, 0.0)).values())
    assert all(diff2.map(lambda v: np.allclose(v, 0.0)).values())

def test_115_116():
    alg = Algebra(3, 0, 1)
    N = 500
    x = np.random.random_sample(N)
    y = np.random.random_sample(N)
    z = np.random.random_sample(N)
    points = alg.vector(e0=np.ones(N), e1=x, e2=y, e3=z).dual()

    assert points.shape == (N,)
    assert len(points) == N

    i = 0
    for point in points:
        i += 1
    assert i == N

    point = points[0]
    assert point.shape == ()
    with pytest.raises(TypeError): len(point)
    assert bool(point) == True
    with pytest.raises(TypeError): iter(point)

    empty = alg.multivector()
    assert empty.shape == ()
    with pytest.raises(TypeError): len(empty)
    assert bool(empty) == False
    with pytest.raises(TypeError): iter(empty)
