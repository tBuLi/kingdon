from collections import OrderedDict

import pytest

from kingdon import (
    Algebra,
    Scalar, Vector, Bivector, Trivector, Quadvector, Pentavector, Hexavector, Heptavector, Octovector, # k-vectors
    Bireflection,
)
from kingdon.multivector import (
    MultiVector, Direction, EVector,
    UPoint, Point, Translation,
    KVector,
)
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import resolve_layout

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
    return tuple(g % (alg.d + 1) for g in grds if (-alg.d - 1) <= g <= alg.d)

@pytest.mark.parametrize(
    "MVType, layout, grades, bases",
    [
        (Scalar, {0: ...}, (0,), (KVector,)),
        (Vector, {1: ..., 2: ..., 4: ..., 8: ...}, (1,), (KVector,)),
        (Bivector, {9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (2,), (KVector,)),
        (Trivector, {14: ..., 13: ..., 11: ..., 7: ...}, (3,), (KVector,)),
        (Quadvector, {15: ...}, (4,), (KVector,)),
        (Direction, {14: ..., 13: ..., 11: ...}, (-2,), (MultiVector,)),
        (EVector, {1: ..., 2: ..., 4: ...}, (1,), (Vector,)),
        (UPoint, {1: ..., 2: ..., 4: ..., 8: 1.0}, (1,), (Vector,)),
        (Point, {14: ..., 13: ..., 11: ..., 7: 1.0}, (-2,), (MultiVector,)),
        (Translation, {0: 1.0, 9: ..., 10: ..., 12: ...}, (0, 2), (Bireflection,)),
        (Bireflection, {0: ..., 9: ..., 10: ..., 12: ..., 3: ..., 5: ..., 6: ...}, (0, 2), (MultiVector,)),
    ],
)
@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_pga_archetypes(alg_name, MVType, layout, grades, bases):
    """ Test if the archetypes correctly generate the expected layout. Done for different PGA's to ensure the validity. """
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
    if max(grades) > alg.d: return

    # These types should have a layout on the algebra.
    alg_layout = alg._type_layouts[MVType]
    assert alg_layout == layout
    assert OrderedDict(alg_layout) == OrderedDict(layout)
    assert issubclass(MVType, bases)

    # Alternative: directly create the archetype. This is the symbolic multivector the
    # layout is derived from, and is untyped.
    x = MVType.archetype(alg, 'x')
    assert type(x) is MultiVector
    assert x.grades == pos_grades(alg, grades)
    assert x.shape == (len(x.keys()),)
    assert x.keys() == tuple(alg_layout)
    assert all([isinstance(a, RationalPolynomial) for a in x.values()])
    assert all(float(str(a)) == b for a, b in zip(x.values(), layout.values()) if b != ...)

    # Fromname should be produced using the layout, and only feature free variables.
    X = MVType.fromname(alg, 'x')
    assert isinstance(X, MVType)
    assert X.grades == pos_grades(alg, grades)
    assert X.shape == (len(X.keys()),)
    assert X.keys() == tuple(k for k, v in alg_layout.items() if v == ...)
    # Every multivector knows the layout of its own type.
    assert X.layout == alg_layout

    # Similarly, __new__ should produce mv's with only the keys allowed by the layout.
    mv = getattr(alg, MVType.__name__.lower())(name='x')
    assert isinstance(mv, MVType)
    assert mv.grades == pos_grades(alg, grades)
    assert mv.shape == (len(mv.keys()),)
    assert mv.keys() == tuple(k for k, v in alg_layout.items() if v == ...)
    assert mv == X


def test_layout(pga3d):
    layout = pga3d._type_layouts[Point]
    keys = tuple(layout)
    with pytest.raises(TypeError):  # The origin key is not in the free part of the layout.
        pga3d.point(name='x', keys=keys)
    # Instead, we are only allowed to provide keys in the free part of the layout.
    keys = tuple(k for k, v in layout.items() if v == ...)
    p = pga3d.point(name='x', keys=keys)
    assert p == pga3d.point(name='x')
    assert str(p) == 'x032 𝐞₀₃₂ + x013 𝐞₀₁₃ + x021 𝐞₀₂₁ + 1.0 𝐞₁₂₃'

    # The origin is the special case of a point with no keys.
    origin = pga3d.point(name='x', keys=())
    assert origin.keys() == ()
    assert origin.values() == []
    assert origin.shape == (0,)
    assert str(origin) == '1.0 𝐞₁₂₃'

    # Let's also test a point with a subset of keys.
    pz = pga3d.point(e012=3)
    assert isinstance(pz, Point)
    assert pz.keys() == (11,)  # e021 = 8 + 2 + 1
    assert pz.values() == [-3]  # e021 = -e012.
    assert pz.shape == (1,)
    assert str(pz) == '-3 𝐞₀₂₁ + 1.0 𝐞₁₂₃'

    xyz = ['x', 'y', 'z']
    pxyz = pga3d.point(xyz)  # Verify that the constructor is in the expected xyz order.
    assert str(pxyz) == 'x 𝐞₀₃₂ + y 𝐞₀₁₃ + z 𝐞₀₂₁ + 1.0 𝐞₁₂₃'



# TODO: add 2d and 3d pga parameterizations.
@pytest.mark.parametrize("MVType, DType, UDType, grades, dgrades", [
    (Direction, EVector, EVector, (-2,), (1,)),
    (EVector, Direction, Direction, (1,), (-2,)),
    (Point, Vector, UPoint, (-2,), (1,)),
    (UPoint, Point, Trivector, (1,), (-2,)),
    (Vector, Trivector, Trivector, (1,), (3,)),
    (Trivector, Vector, Vector, (3,), (1,)),
])
def test_pga_duality_relations(MVType, DType, UDType, grades, dgrades):
    """ Test if the dual and undual of various types are the expected type. """
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
    assert q_reversed.grades == (alg.d - 1,)
    assert q_reversed.shape == (alg.d,)  # x y (z) w are free variables for a PseudoVector.

    # The product of a point and a pseudovector is a bireflection, because this path cannot determine the constraint on the scalar part.
    t = p * q_reversed
    assert isinstance(t, Bireflection)
    assert t.grades == (0, 2)
    assert t.shape == (alg.d,)  # x y (z) w are free variables for a Bireflection.

    # However, we know it should really be a translation, which can be achieved by compiling the same scenario.
    @alg.jit(symbolic=True)
    def translate(p, q):
        return p * q.reverse()
    t = translate(p, q)
    assert isinstance(t, Translation)
    assert t.grades == (0, 2)
    assert t.shape == (alg.d - 1,)  # Only x y (z) are free variables.
    layout = t.layout
    assert layout == alg._type_layouts[Translation]
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
    """ Test the resolve layout function. """
    mvtype, layout = resolve_layout(layouts, res_layout)
    if expected is None:
        assert (mvtype, layout) == (MultiVector, {})
    else:
        assert mvtype == expected


@pytest.mark.parametrize(
    "source_type, target_type",
    [
        # Free key in source conflicts with a fixed (normalisation) key in target.
        (Vector,      UPoint),
        (Trivector, Point),
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
    "source_type, target_type, target_grades, expected_keys, expected_fixed",
    [
        (Scalar,       MultiVector,  (0,),   (0,),                     {}),
        (Vector,       MultiVector,  (1,),   (1, 2, 4, 8),             {}),
        (Bivector,     MultiVector,  (2,),   (9, 10, 12, 3, 5, 6),     {}),
        (Trivector,    MultiVector,  (3,),  (14, 13, 11, 7),          {}),
        (Direction,    MultiVector,  (3,),  (14, 13, 11),             {}),
        (EVector,      MultiVector,  (1,),   (1, 2, 4),                {}),
        (UPoint,       MultiVector,  (1,),   (1, 2, 4, 8),             {8: 1.0}),
        (Point,        MultiVector,  (3,),  (14, 13, 11, 7),          {7: 1.0}),
        (Translation,  MultiVector,  (0, 2),  (0, 9, 10, 12),           {0: 1.0}),
        (Bireflection, MultiVector,  (0, 2),  (0, 9, 10, 12, 3, 5, 6),  {}),
        (Point,        Trivector, (3,),  (14, 13, 11, 7),          {7: 1.0}),
        (Direction,    Trivector, (3,),  (14, 13, 11),             {}),
        (UPoint,       Vector,       (1,),   (1, 2, 4, 8),             {8: 1.0}),
        (EVector,      Vector,       (1,),   (1, 2, 4),                {}),
        (Translation,  Bireflection,  (0, 2),  (0, 9, 10, 12),           {0: 1.0}),
    ],
)
def test_asmvtype(pga3d, source_type, target_type, target_grades, expected_keys, expected_fixed):
    """ Test conversion of multivector types using the asmvtype method. """
    alg = pga3d
    source = source_type.fromname(alg, 'x')
    result = source.asmvtype(target_type)

    assert isinstance(result, target_type)
    assert result is not source
    assert result.grades == target_grades
    assert result.keys() == expected_keys
    for k, v in expected_fixed.items():
        assert getattr(result, alg.bin2canon[k]) == v
    # Free values carried from source must survive the conversion.
    for k in result.keys():
        blade = pga3d.bin2canon[k]
        assert getattr(result, blade) == getattr(source, blade)


@pytest.mark.parametrize("T1, T2, T3, func", [
    (Point, Direction, Point, lambda p, q: p + q),
    (Point, Point, Direction, lambda p, q: p - q),
    (Point, Direction, (-3,), lambda p, q: p & q),
    (Point, Point, (-3,), lambda p, q: p & q),
    (Vector, Point, (-2,), lambda p, q: p >> q),  # Reflection of a point is a -point.
    (Translation, Point, Point, lambda p, q: p >> q),
    (Bireflection, Point, Point, lambda p, q: p >> q),
    # Test twisted Lipschitz action for orthogonal transformations.
    (Vector, Vector, Vector, lambda p, q: p >> q),
    (Vector, Bivector, Bivector, lambda p, q: p >> q),
    (Vector, Trivector, Trivector, lambda p, q: p >> q),
    (Bireflection, Vector, Vector, lambda p, q: p >> q),
    (Bireflection, Bireflection, Bireflection, lambda p, q: p >> q),
    # Point reflection should preserve bireflections
    (Point, Translation, Translation, lambda p, q: p >> q),
    (Point, Bireflection, Bireflection, lambda p, q: p >> q),
    # Commutator is grade preserving, although not always type preserving
    (Bivector, Vector, Vector, lambda p, q: p.cp(q)),
    (Bivector, Bivector, {'2DPGA': Direction, '3DPGA': Bivector}, lambda p, q: p.cp(q)),
    (Bivector, Trivector, {'2DPGA': Scalar, '3DPGA': Direction}, lambda p, q: p.cp(q)),
    (Bivector, Point, Direction, lambda p, q: p.cp(q)),
    (Bivector, Bireflection, {'2DPGA': Direction, '3DPGA': Bivector}, lambda p, q: p.cp(q)),
    (Bivector, Translation, {'2DPGA': Direction, '3DPGA': Bivector}, lambda p, q: p.cp(q)),
])
@pytest.mark.parametrize("alg_name", ['2DPGA', '3DPGA'])
def test_type2type(alg_name, T1, T2, T3, func):
    """ Test if type's are preserved under operators the way they should be """
    alg = Algebra.fromname(alg_name)
    t1 = T1.fromname(alg, 'x')
    t2 = T2.fromname(alg, 'y')
    result = func(t1, t2)
    if isinstance(T3, tuple):
        assert result.grades == pos_grades(alg, T3)
    else:
        if isinstance(T3, dict):
            T3 = T3[alg_name]
        assert isinstance(result, T3)


@pytest.mark.parametrize("T, grade", [
    (Scalar, 0),
    (Vector, 1),
    (Bivector, 2),
    (Trivector, 3),
    (Quadvector, 4),
    (Pentavector, 5),
    (Hexavector, 6),
    (Heptavector, 7),
    (Octovector, 8),
])
@pytest.mark.parametrize("dim", [4, 8, 9])
def test_grade_selection(dim, T, grade):
    alg = Algebra(dim)
    x = alg.multivector(name='x')
    xg = x.grade(grade)
    if grade > dim:
        assert type(xg) == MultiVector
    else:
        assert type(xg) == T

def test_custom_types():
    """ Allow a user make their own types. """
    class MyScalar(Scalar):
        pass

    # Use only custom types
    mytypes = [MyScalar]
    alg = Algebra(9, types=mytypes)
    assert alg.types == mytypes
    assert type(alg.blades.e) is MyScalar

    # Extend default types with custom types.
    default_types = Algebra(9).types
    alg = Algebra(9, extra_types=mytypes)
    assert alg.types == [*default_types, *mytypes]

def test_constructors(pga2d):
    """ Every registered type gets a constructor on the algebra. """
    alg = pga2d
    kvectors = [Scalar, Vector, Bivector, Trivector]

    # Every k-vector type has a constructor of the matching grade.
    for grade, T in enumerate(kvectors):
        x = getattr(alg, T.__name__.lower())(name='x')
        assert type(x) == T
        assert x.grades == (grade,)

    # A pseudo-k-vector is the dual of a k-vector, and hence of grade d - k.
    for grade, T in enumerate(kvectors):
        px = getattr(alg, f'pseudo{T.__name__.lower()}')(name='x')
        assert type(px) == kvectors[alg.d - grade]
        assert px.grades == (alg.d - grade,)

    assert type(alg.pseudoscalar(name='x')) == Trivector
    assert type(alg.pseudovector(name='x')) == Bivector

    # The PGA types are only registered in algebras where they make sense.
    for attr, T in [('point', Point), ('upoint', UPoint), ('direction', Direction),
                    ('evector', EVector), ('translation', Translation),
                    ('bireflection', Bireflection)]:
        assert type(getattr(alg, attr)(name='x')) == T
    assert not hasattr(Algebra(2), 'point')
