from collections import OrderedDict

import pytest

from kingdon import Algebra, Scalar, PseudoScalar, Vector, Bivector, PseudoVector, PseudoBivector, Bireflection
from kingdon.multivector import (
    MultiVector, Direction, EVector,
    UPoint, Point, Translation,
)
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import LayoutResolver

@pytest.fixture
def vga2d():
    return Algebra(2)

@pytest.fixture
def pga2d():
    return Algebra.fromname('2DPGA')

@pytest.fixture
def pga3d():
    return Algebra.fromname('3DPGA')

def pos_grades(alg, grds):
    return tuple(g % (alg.d + 1) for g in grds)

@pytest.mark.parametrize(
    "MVType, layout, grades, bases",
    [
        (Scalar, {0: ...}, (0,), (MultiVector,)),
        (Vector, {1: ..., 2: ..., 4: ..., 8: ...}, (1,), (MultiVector,)),
        (Bivector, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (MultiVector,)),
        (PseudoBivector, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (-3,), (MultiVector,)),
        (PseudoVector, {14: ..., 13: ..., 11: ..., 7: ...}, (-2,), (MultiVector,)),
        (PseudoScalar, {15: ...}, (-1,), (MultiVector,)),
        (Direction, {14: ..., 13: ..., 11: ...}, (-2,), (PseudoVector,)),
        (EVector, {1: ..., 2: ..., 4: ...}, (1,), (Vector,)),
        (UPoint, {1: ..., 2: ..., 4: ..., 8: 1.0}, (1,), (Vector,)),
        (Point, {14: ..., 13: ..., 11: ..., 7: 1.0}, (-2,), (PseudoVector,)),
        (Translation, {0: 1.0, 9: ..., 10: ..., 12: ...}, (0, 2), (MultiVector,)),
        (Bireflection, {0: ..., 9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (0, 2), (MultiVector,)),
    ],
)
@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_pga_archetypes(alg_name, MVType, layout, grades, bases):
    alg = Algebra.fromname(alg_name)
    if alg_name == '2DPGA':
        # 3DPGA key bits: {0=e1, 1=e2, 2=e3, 3=e0}. 2DPGA key bits: {0=e1, 1=e2, 2=e0}.
        # Discard keys with e3 (bit 2 in 3DPGA), remap e0 from 3DPGA bit 3 → 2DPGA bit 2.
        def direct(k):
            return None if (k & 4) else (k & 3) | ((k >> 1) & 4)
        if all(g < 0 for g in grades):
            # Pseudo/dual types: complement in 3D → direct remap → complement in 2D.
            def remap(k):
                p = direct(15 - k)
                return None if p is None else 7 - p
        else:
            remap = direct
        remapped = {k2: v for k, v in layout.items() if (k2 := remap(k)) is not None}
        layout = {k: remapped[k] for k in alg.canon2bin.values() if k in remapped}
    archetype = alg.archetypes[MVType]
    assert archetype.layout == layout
    assert OrderedDict(archetype.layout) == OrderedDict(layout)
    assert issubclass(MVType, bases)

    x = MVType.archetype(alg, 'x')
    assert isinstance(x, MVType)
    assert x.grades == pos_grades(alg, grades)
    assert x.shape == (len(x.keys()),)
    assert x.keys() == tuple(archetype.layout)
    assert all([isinstance(a, RationalPolynomial) for a in x.values()])
    assert all(float(str(a)) == b for a, b in zip(x.values(), layout.values()) if b != ...)
    # Fromname should be produced using the layout, and only feature free variables.
    X = MVType.fromname(alg, 'x')
    assert isinstance(X, MVType)
    assert X.grades == pos_grades(alg, grades)
    assert X.shape == (len(X.keys()),)
    assert X.keys() == tuple(k for k, v in archetype.layout.items() if v == ...)
    # Similarly, __new__ should produce mv's with only the keys allowed by the layout.
    mv = getattr(alg, MVType.__name__.lower())(name='x')
    assert isinstance(mv, MVType)
    assert mv.grades == pos_grades(alg, grades)
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


# TODO: add 2d and 3d pga parameterizations.
@pytest.mark.parametrize("MVType, DType, UDType, grades, dgrades", [
    (Direction, EVector, EVector, (-2,), (1,)),
    (EVector, Direction, Direction, (1,), (-2,)),
    (Point, Vector, UPoint, (-2,), (1,)),
    (UPoint, Point, PseudoVector, (1,), (-2,)),
    (Vector, PseudoVector, PseudoVector, (1,), (3,)),
    (PseudoVector, Vector, Vector, (3,), (1,)),
])
def test_pga_duality_relations(MVType, DType, UDType, grades, dgrades):
    alg = Algebra.fromname('3DPGA')
    mv = MVType.fromname(alg, 'x')
    assert mv.grades == pos_grades(alg, grades)
    assert isinstance(mv, MVType)

    dmv = mv.dual()
    assert isinstance(dmv, DType)

    udmv = mv.undual()
    assert dmv.grades == pos_grades(alg, dgrades)
    assert isinstance(udmv, UDType)

@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_translations(alg_name):
    alg = Algebra.fromname(alg_name)
    p = alg.point(name='p')
    q = alg.point(name='q')
    q_reversed = q.reverse()
    assert isinstance(q_reversed, PseudoVector)
    assert q_reversed.grades == (alg.d - 1,)
    assert q_reversed.shape == (alg.d,)  # x y (z) w are free variables for a PseudoVector.

    # The product of a point and a pseudovector is a bireflection.
    t = p * q_reversed
    assert isinstance(t, Bireflection)
    assert t.grades == (0, 2)
    assert t.shape == (alg.d,)  # x y (z) w are free variables for a Bireflection.

    # However, we know it should really be a translation, which can be achieved by compiling the same scenario.
    @alg.compile(symbolic=True)
    def translate(p, q):
        return p * q.reverse()
    t = translate(p, q)
    assert isinstance(t, Translation)
    assert t.grades == (0, 2)
    assert t.shape == (alg.d - 1,)  # Only x y (z) are free variables.
    layout = alg.archetypes[Translation].layout
    assert all(layout[k] == ... for k in t.keys())
    assert t.e == 1.0  # While this is not one of the free variables, it is retrieved from the layout.

    diff = (p - q).undual()
    assert isinstance(p - q, Direction) and isinstance(diff, EVector)
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
        # reject layout with no entry for a fixed key in res (e.g. Direction vs Point when e_w=1.0 is present)
        ({14: ..., 13: ..., 11: ..., 7: 1.0},
         {'Direction': {14: ..., 13: ..., 11: ...}, 'Point': {14: ..., 13: ..., 11: ..., 7: 1.0}},
         'Point'),
    ],
)
def test_best_fit_layout_cases(res_layout, layouts, expected):
    mvtype, layout = LayoutResolver(layouts).resolve(res_layout)
    if expected is None:
        assert (mvtype, layout) == (None, None)
    else:
        assert mvtype == expected


@pytest.mark.parametrize(
    "source_type, target_type",
    [
        # Free key in source conflicts with a fixed (normalisation) key in target.
        (Vector,      UPoint),
        (PseudoVector, Point),
        (Scalar,      Translation),
        (Bireflection, Translation),
        # Source keys absent from the target layout entirely (grade mismatch).
        (Scalar,    Vector),
        (Vector,    Scalar),
        (Point,     Direction),
        (UPoint,    EVector),
        (Bivector,  Scalar),
    ],
)
def test_asmvtype_incompatible(pga3d, source_type, target_type):
    """asmvtype raises TypeError when casting to an incompatible layout."""
    source = source_type.fromname(pga3d, 'x')
    with pytest.raises(TypeError):
        source.asmvtype(target_type)


@pytest.mark.parametrize(
    "source_type, target_type, expected_keys, expected_fixed",
    [
        (Scalar,       MultiVector,  (0,),                     {}),
        (Vector,       MultiVector,  (1, 2, 4, 8),             {}),
        (Bivector,     MultiVector,  (9, 10, 12, 3, 5, 6),     {}),
        (PseudoVector, MultiVector,  (14, 13, 11, 7),          {}),
        (PseudoScalar, MultiVector,  (15,),                    {}),
        (Direction,    MultiVector,  (14, 13, 11),             {}),
        (EVector,      MultiVector,  (1, 2, 4),                {}),
        (UPoint,       MultiVector,  (1, 2, 4, 8),             {8: 1.0}),
        (Point,        MultiVector,  (14, 13, 11, 7),          {7: 1.0}),
        (Translation,  MultiVector,  (0, 9, 10, 12),           {0: 1.0}),
        (Bireflection,  MultiVector,  (0, 9, 10, 12, 3, 5, 6),  {}),
        (Point,        PseudoVector, (14, 13, 11, 7),          {7: 1.0}),
        (Direction,    PseudoVector, (14, 13, 11),             {}),
        (UPoint,       Vector,       (1, 2, 4, 8),             {8: 1.0}),
        (EVector,      Vector,       (1, 2, 4),                {}),
        (Bivector,     PseudoBivector, (9, 10, 12, 3, 5, 6),  {}),
        (PseudoBivector, Bivector,   (9, 10, 12, 3, 5, 6),    {}),
        (Translation,  Bireflection,  (0, 9, 10, 12),           {0: 1.0}),
    ],
)
def test_asmvtype(pga3d, source_type, target_type, expected_keys, expected_fixed):
    """ Test conversion of multivector types using the asmvtype method. """
    alg = pga3d
    source = source_type.fromname(alg, 'x')
    result = source.asmvtype(target_type)

    assert isinstance(result, target_type)
    assert result is not source
    assert result.keys() == expected_keys
    for k, v in expected_fixed.items():
        assert getattr(result, alg.bin2canon[k]) == v
    # Free values carried from source must survive the conversion.
    for k in result.keys():
        blade = pga3d.bin2canon[k]
        assert getattr(result, blade) == getattr(source, blade)


@pytest.mark.parametrize("MVType, MVType_alt, grades", [
    (Bivector, Vector ^ Vector, (2,)),
    (Bireflection, Vector * Vector, (0, 2,)),
    (PseudoBivector, PseudoVector & PseudoVector, (-3,)),
    (Translation, Point * ~Point, (0, 2)),
])
def test_mvtype_cache(MVType, MVType_alt, grades):
    """ It is possible to construct MVType's structurally rather than referencing them by name. """
    assert MVType is MVType_alt
    assert MVType.grades == grades
    assert MVType_alt.grades == grades


@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_codimension(alg_name):
    """ In PGA it is important to distinguish dimension from co-dimension. """
    alg = Algebra.fromname(alg_name)

    p, q = alg.point(name='p'), alg.point(name='q')
    l = p & q
    assert isinstance(l, PseudoBivector)

    a, b = alg.upoint(name='a'), alg.upoint(name='b')
    ab = a ^ b
    assert isinstance(ab, Bivector)

    assert isinstance(ab.dual(), PseudoBivector)
    assert isinstance(ab.undual(), PseudoBivector)
