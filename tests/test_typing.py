from collections import OrderedDict

import pytest

from kingdon import Algebra, Scalar, Vector, Bivector, PseudoVector, PseudoBivector, PseudoScalar
from kingdon.multivector import (
    MultiVector, Blade, Blade2, Direction, EVector, 
    UPoint, Point, DPoint, DDPoint, Translation, Line, Reflection2
)
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import best_fit_layout

@pytest.fixture
def vga2d():
    return Algebra(2)

@pytest.fixture
def pga2d():
    return Algebra.fromname('2DPGA')

@pytest.fixture
def pga3d():
    return Algebra.fromname('3DPGA')


@pytest.mark.parametrize(
    "MVType, layout, grades, bases",
    [
        (Scalar, {0: ...}, (0,), (Blade,)),
        (Vector, {1: ..., 2: ..., 4: ..., 8: ...}, (1,), (Blade,)),
        (Bivector, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (MultiVector,)),
        (PseudoBivector, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (MultiVector,)),
        (PseudoVector, {14: ..., 13: ..., 11: ..., 7: ...}, (3,), (Blade,)),
        (PseudoScalar, {15: ...}, (4,), (Blade,)),
        (Blade2, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (Blade,)),
        (Direction, {14: ..., 13: ..., 11: ...}, (3,), (Blade,)),
        (EVector, {1: ..., 2: ..., 4: ...}, (1,), (Blade,)),
        (UPoint, {1: ..., 2: ..., 4: ..., 8: 1.0}, (1,), (Blade,)),
        (Point, {14: ..., 13: ..., 11: ..., 7: 1.0}, (3,), (Blade,)),
        (Translation, {0: 1.0, 9: ..., 10: ..., 12: ...}, (0, 2), (MultiVector,)),
        (Line, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (Blade,)),
        (Reflection2, {0: ..., 9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (0, 2), (MultiVector,)),
        # Include bireflections, rotations and translations etc.
    ],
)
def test_pga_archetypes(pga3d, MVType, layout, grades, bases):
    # TODO: extend this test to pga2d. The layout can be obtained by filtering the 3d layout with 0x1011.
    archetype = pga3d.archetypes[MVType]
    assert archetype.layout == layout
    assert OrderedDict(archetype.layout) == OrderedDict(layout)
    assert issubclass(MVType, bases)

    x = MVType.archetype(pga3d, 'x')
    assert isinstance(x, MVType)
    assert x.grades == grades
    assert x.shape == (len(x.keys()),)
    assert x.keys() == tuple(archetype.layout)
    assert all([isinstance(a, RationalPolynomial) for a in x.values()])
    assert all(float(str(a)) == b for a, b in zip(x.values(), layout.values()) if b != ...)
    # Fromname should be produced using the layout, and only feature free variables.
    X = MVType.fromname(pga3d, 'x')
    assert isinstance(X, MVType)
    assert X.grades == grades
    assert X.shape == (len(X.keys()),)
    assert X.keys() == tuple(k for k, v in archetype.layout.items() if v == ...)
    # Similarly, __new__ should produce mv's with only the keys allowed by the layout.
    mv = getattr(pga3d, MVType.__name__.lower())(name='x')
    assert isinstance(mv, MVType)
    assert mv.grades == grades
    assert mv.shape == (len(mv.keys()),)
    assert mv.keys() == tuple(k for k, v in archetype.layout.items() if v == ...)
    assert mv == X


def test_layout(pga3d):
    archetype = pga3d.archetypes[Point]
    keys = tuple(archetype.layout)
    with pytest.raises(TypeError):  # The origin key is not in the free part of the layout.
        pga3d.point(name='x', keys=keys)
    # Instead, we are only allowed to provide keys in the free part of the layout.
    keys = tuple(k for k, v in archetype.layout.items() if v == ...)
    assert pga3d.point(name='x', keys=keys) == pga3d.point(name='x')
    # The origin is the special case of a point with no keys.
    origin = pga3d.point(name='x', keys=())
    assert origin.keys() == ()
    assert origin.values() == []
    assert origin.shape == (0,)
    # Let's also text a point with a subset of keys.
    pz = pga3d.point(e012=3)
    assert isinstance(pz, Point)
    assert pz.keys() == (11,)  # e021 = 8 + 2 + 1
    assert pz.values() == [-3]  # e021 = -e012.
    assert pz.shape == (1,)


def test_blade_calculations(pga3d):
    alg = pga3d
    b = Blade2.fromname(alg, 'uv')
    B = Bivector.fromname(alg, 'B')
    assert isinstance(b, Blade2)
    # assert isinstance(b, Bivector)  # TODO: this should work?
    assert b.grades == (2,)
    assert B.grades == (2,)
    # For a 2-blade b, b**2 is a scalar. However, for a bivector B, it is not.
    b_sq = b**2
    assert b_sq.grades == (0,)
    assert isinstance(b_sq, Scalar)
    B_sq = B**2
    assert B_sq.grades == (0, 4)
    # assert isinstance(B_sq, Study)

    u = Vector.fromname(alg, 'u')
    v = Vector.fromname(alg, 'v')
    uv = u ^ v
    assert isinstance(uv, Blade2)
    assert uv.grades == (2,)


# TODO: add 2d and 3d pga parameterizations.
@pytest.mark.parametrize("MVType, DType, UDType, grades, dgrades", [
    (Direction, EVector, EVector, (-2,), (1,)),
    (EVector, Direction, Direction, (1,), (-2,)),
    (Point, DPoint, UPoint, (-2,), (1,)),
    (UPoint, Point, DDPoint, (1,), (-2,)),
    (Vector, PseudoVector, PseudoVector, (1,), (3,)),
    (PseudoVector, Vector, Vector, (3,), (1,)),
])
def test_pga_duality_relations(MVType, DType, UDType, grades, dgrades):
    def pos_grades(grds):
        return tuple(g % (alg.d + 1) for g in grds)
    alg = Algebra.fromname('3DPGA')
    mv = MVType.fromname(alg, 'x')
    # assert mv.grades == pos_grades(grades)
    assert isinstance(mv, MVType)

    dmv = mv.dual()
    assert isinstance(dmv, DType)
    assert mv == dmv.undual()
    assert mv == - dmv.dual()

    udmv = mv.undual()
    # assert dmv.grades == pos_grades(dgrades)
    assert isinstance(udmv, UDType)
    assert mv == udmv.dual()
    assert mv == - udmv.undual()  # Will work once we have proper type inference.

@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_translations(alg_name):
    alg = Algebra.fromname(alg_name)
    p = alg.point(name='p')
    q = alg.point(name='q')
    q_reversed = q.reverse()
    assert q_reversed.grades == (alg.d - 1,)
    assert q_reversed.shape == (alg.d - 1,)  # Only x y (z) are free variables.
    assert not isinstance(q_reversed, Point) and isinstance(q_reversed.reverse(), Point)  # Is an instance of the temporary ReversePoint class.
    t = p * q_reversed
    assert isinstance(t, Translation)
    assert t.grades == (0, 2)
    assert t.shape == (alg.d - 1,)  # Only x y (z) are free variables.
    layout = alg.archetypes[Translation].layout
    assert all(layout[k] == ... for k in t.keys())
    
    diff = (p - q).undual()
    assert isinstance(p - q, Direction) and isinstance(diff, EVector)
    assert t.e == 1.0  # While this is not one of the free variables, it is retrieved from the layout.
    assert not (t - diff*alg.blades.e0)

@pytest.mark.parametrize(
    "res_layout, layouts, expected",
    [
        # exact float beats loose
        ({1: ..., 2: ..., 8: 1.0},
         {'PseudoVector': {1: ..., 2: ..., 8: ...}, 'Point': {1: ..., 2: ..., 8: 1.0}},
         'Point'),
        # among exact matches, least excess wins
        ({1: ..., 2: ...},
         {'Big': {1: ..., 2: ..., 4: ..., 8: ...}, 'Small': {1: ..., 2: ...}, 'Mid': {1: ..., 2: ..., 4: ...}},
         'Small'),
        # reject fixed values outside res_layout
        ({1: ..., 2: ...},
         {'WithStrayFixed': {1: ..., 2: ..., 4: 0.0}, 'Free': {1: ..., 2: ..., 4: ...}},
         'Free'),
        # reject missing/non-free ellipsis slot
        ({1: ..., 2: ..., 4: ...},
         {'NoSlot4': {1: ..., 2: ...}, 'Fixed4': {1: ..., 2: ..., 4: 5.0}, 'Ok': {1: ..., 2: ..., 4: ...}},
         'Ok'),
        # reject conflicting fixed float
        ({1: ..., 8: 1.0},
         {'Wrong': {1: ..., 8: 2.0}, 'Right': {1: ..., 8: 1.0}},
         'Right'),
        # no feasible candidate
        ({1: ..., 8: 1.0},
         {'Wrong': {1: ..., 8: 2.0}, 'Stray': {1: ..., 8: 1.0, 4: 3.0}},
         None),
        # loose cost dominates excess cost
        ({1: ..., 8: 1.0},
         {'LooseTight': {1: ..., 8: ...}, 'ExactBigger': {1: ..., 8: 1.0, 4: ..., 2: ...}},
         'ExactBigger'),
        # tie breaks by insertion order
        ({1: ..., 2: ...},
         {'First': {1: ..., 2: ...}, 'Second': {1: ..., 2: ...}},
         'First'),
    ],
)
def test_best_fit_layout_cases(res_layout, layouts, expected):
    mvtype, layout = best_fit_layout(res_layout, layouts)
    if expected is None:
        assert (mvtype, layout) == (None, None)
    else:
        assert mvtype == expected


# Workflow:
# - Archetype
# - Layout
# - Validate against layout in constructors, including in fromname.
# - codegen should use archetype instead of fromname to ensure a correct symbolic mv.
# - if possible, bootstrap the whole thing from the archetypes, even while using codegen in the definitions of the archetypes.
# - Printing to strings should use the layout, not the keys.
# - verify that type is considered in comparison even though layout is not.
# - Perhaps "intermediate" types like Blade2 should not be available as a constructor?
# - TODO: pre-sort the layouts in algebra.type_layouts to make best_fit_layout faster.
# - CSE test for (a & b & c) * ~(a & b & c) https://bivector.net/CLEANUP.html

# Things to test:
# - Translation should always have scalar 1 and this is not a free variable, but the grade should still be shown as (0, 2).
# Reverse of a point is not a point perse, be sure to test against this bug.