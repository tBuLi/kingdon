import numpy as np
import pytest
from sympy import Matrix, Symbol

from kingdon import Algebra
from kingdon.matrixreps import expr_as_matrix
from kingdon.multivector import Point
from kingdon.taperecorder import TapeRecorder


def assert_string_keys(*mvs):
    for mv in mvs:
        assert isinstance(mv.keys(), tuple)
        assert all(
            isinstance(key, str) and key.startswith('e') for key in mv.keys()
        )
        assert all(isinstance(key, str) for key, _ in mv.items())


def test_string_keys_are_canonical_for_construction_access_and_layouts():
    alg = Algebra(3)
    for name in ('blade2mask', 'mask2blade', 'canon2bin', 'bin2canon', 'signs'):
        assert not hasattr(alg, name)
    symbolic = alg.multivector(name='x', keys=('e', 'e1', 'e23'))
    numeric = alg.multivector({'e': 1, 'e1': 2, 'e23': 3})
    assert symbolic.keys() == numeric.keys()
    assert symbolic.keys() == ('e', 'e1', 'e23')
    assert symbolic.e == Symbol('x')
    assert symbolic.e23 == Symbol('x23')
    assert dict(numeric.items()) == {'e': 1, 'e1': 2, 'e23': 3}
    assert 'e23' in numeric and 6 not in numeric
    assert_string_keys(
        symbolic, numeric
    )

    with pytest.raises(KeyError):
        alg.multivector(keys=(0, 1), values=[1, 2])
    with pytest.raises(KeyError):
        alg.multivector({0: 1})
    with pytest.raises(TypeError):
        type(numeric).fromkeysvalues(alg, (0,), [1])

    vector = alg.vector(name='v')
    assert vector.type_layout == {'e1': ..., 'e2': ..., 'e3': ...}
    assert all(
        isinstance(key, str)
        for layout in alg._type_layouts.values()
        for key in layout
    )


@pytest.mark.parametrize('basis, message', [
    (['e', 'e1', 'e2'], 'every blade'),
    (['e', 'e1', 'e12', 'e21'], 'every generator blade'),
    (['e', 'e1', 'e2', 'e1'], 'Duplicate generator support'),
    (['e', 'e1', 'e2', 'e11'], 'Invalid custom basis blade'),
    (['e', 'e1', 'e2', 'e1@'], 'Invalid custom basis blade'),
    (['e', 'e12', 'e1', 'e2'], 'grade order'),
])
def test_custom_basis_validation(basis, message):
    with pytest.raises(ValueError, match=message):
        Algebra(2, basis=basis)


def test_grade_requests_follow_basis_order():
    default = Algebra(3)
    expected = ('e1', 'e2', 'e3', 'e12', 'e13', 'e23')
    assert tuple(default.indices_for_grades((2, 1))) == expected
    assert tuple(default.indices_for_grades((1, 2))) == expected

    custom = Algebra.fromname('3DPGA')
    expected = ('e1', 'e2', 'e3', 'e0', 'e01', 'e02', 'e03', 'e12', 'e31', 'e23')
    assert tuple(custom.indices_for_grades((2, 1))) == expected
    assert tuple(custom.indices_for_grades((1, 2))) == expected


def test_named_3dpga_full_subset_order():
    algebra = Algebra.fromname('3DPGA', large=True)
    assert algebra.blades.e.asfullmv(canonical=False).keys() == (
        'e', 'e1', 'e2', 'e12', 'e3', 'e31', 'e23', 'e123',
        'e0', 'e01', 'e02', 'e021', 'e03', 'e013', 'e032', 'e0123',
    )


def test_layout_position_compatibility_outputs():
    default = Algebra(3)
    mv = default.multivector(keys=('e', 'e3', 'e13', 'e123'), values=[1, 2, 3, 4])
    assert mv.type_number == 0b10101001
    assert format(mv, 'keys_binary') == '10101001'

    custom = Algebra.fromname('3DPGA')
    mv = custom.multivector(keys=('e31', 'e032'), values=[2, 3])
    assert mv.type_number == 0b0000101000000000
    assert format(mv, 'keys_binary') == '0000101000000000'


def test_vga_products_use_string_keys():
    alg = Algebra(3)
    e1, e2, e12, e23 = (
        alg.blades[key] for key in ('e1', 'e2', 'e12', 'e23')
    )

    gp = e1 * e2
    op = e1 ^ e2
    ip = e1 | e1
    lc = e1.lc(e12)
    rc = e12.rc(e2)
    rp = e12 & e23
    projected = (alg.blades.e + e1 + e12).grade(0, 2)
    dual = e1.dual()

    assert gp == op == e12
    assert ip == alg.blades.e
    assert lc == e2 and rc == e1 and rp == e2
    assert projected.keys() == ('e', 'e12')
    assert dual.keys() == ('e23',) and dual.undual() == e1
    assert_string_keys(gp, op, ip, lc, rc, rp, projected, dual)


@pytest.mark.parametrize('name', ['2DPGA', '3DPGA'])
def test_degenerate_pga_string_keys_and_typed_layouts(name):
    alg = Algebra.fromname(name)
    assert not alg.blades.e0 * alg.blades.e0

    point = alg.upoint(name='p').dual()
    assert isinstance(point, Point)
    assert not (point.dual().undual() - point)
    assert_string_keys(point, point.dual(), alg.blades.e0 * alg.blades.e0)
    assert all(isinstance(key, str) for key in point.type_layout)


def test_named_3dpga_preserves_nonlexicographic_blade_identity_and_signs():
    alg = Algebra.fromname('3DPGA')
    assert list(alg.indices_for_grades(tuple(range(alg.d + 1)))) == [
        'e', 'e1', 'e2', 'e3', 'e0',
        'e01', 'e02', 'e03', 'e12', 'e31', 'e23',
        'e032', 'e013', 'e021', 'e123', 'e0123',
    ]
    assert alg.blades.e3 * alg.blades.e1 == alg.blades.e31
    assert alg.blades.e1 * alg.blades.e3 == -alg.blades.e31
    assert alg.blades.e3 ^ alg.blades.e1 == alg.blades.e31
    assert alg.blades.e3.lc(alg.blades.e31) == alg.blades.e1
    assert alg.blades.e31.rc(alg.blades.e1) == alg.blades.e3
    assert (
        alg.blades.e0 ^ alg.blades.e3 ^ alg.blades.e2
    ) == alg.blades.e032
    assert (
        alg.blades.e0 ^ alg.blades.e2 ^ alg.blades.e1
    ) == alg.blades.e021
    assert alg.multivector(e13=2).keys() == ('e31',)
    assert alg.multivector(e13=2).e31 == -2
    assert alg.blades.e032 & alg.blades.e021 == -alg.blades.e02
    assert alg.blades.e1.hodge() == alg.blades.e032
    assert alg.blades.e31.hodge() == alg.blades.e02
    assert alg.blades.e032.unhodge() == alg.blades.e1

    @alg.add_operator(symbolic=False)
    def reversed_coefficient(x):
        return x.e13

    coefficient = reversed_coefficient(alg.multivector(e31=2))
    assert coefficient.keys() == ('e',)
    assert coefficient.e == -2

    values = np.arange(len(alg))
    full = alg.multivector(values)
    roundtrip = type(full).frommatrix(alg, full.asmatrix())
    assert roundtrip.keys() == full.keys()
    np.testing.assert_allclose(roundtrip.values(), values)

    x = alg.multivector(name='x', keys=('e31', 'e032'))
    symbolic_dummy = alg.scalar(name='s')
    matrix, selected = expr_as_matrix(
        lambda _, value: value,
        symbolic_dummy,
        x,
        res_like=alg.multivector(e31=1),
    )
    assert selected.keys() == ('e31',)
    assert matrix == Matrix([[1, 0]])


def test_codegen_tape_cache_and_type_numbers_use_string_keys():
    for symbolic in (False, True):
        alg = Algebra(3)

        @alg.add_operator(symbolic=symbolic)
        def products(x, y):
            return (x * y).grade(1) + (x ^ y).grade(2)

        x = alg.multivector(e=1, e1=2, e23=3)
        y = alg.multivector(e2=4, e12=5)
        result = products(x, y)
        expected = products.codegen(x, y)
        cache_size = len(products)
        assert products(x, y) == result
        assert len(products) == cache_size
        assert result == expected
        assert_string_keys(result)
        assert all(
            isinstance(key, str)
            for type_key in products
            for _, keys in type_key
            for key in keys
        )

    alg = Algebra(2)
    keys = ('e', 'e2')
    mv = alg.multivector(name='x', keys=keys)
    tape = TapeRecorder.fromname(alg, 'x', keys)
    assert tape.keys() == keys
    assert tape.type_number == mv.type_number
    assert tape.grade(0).keys() == ('e',)


def test_array_empty_and_large_multivectors_have_string_keys():
    alg = Algebra(2)
    array_mv = alg.vector(np.arange(6).reshape(2, 3))
    empty = alg.multivector()
    assert array_mv.keys() == ('e1', 'e2') and array_mv.shape == (3,)
    assert empty.keys() == () and list(empty.items()) == []
    assert_string_keys(array_mv, empty)

    large = Algebra(7, large=True)
    product = large.blades.e1 * large.blades.e2
    assert product.keys() == ('e12',)
    assert_string_keys(product)


def test_sparse_large_does_not_materialize_full_basis():
    algebra = Algebra(20, large=True)
    for name in ('blade2mask', 'mask2blade'):
        assert not hasattr(algebra, name)
    assert len(algebra.blades) == algebra.d + 1  # basis vectors and pseudoscalar
    a = algebra.multivector({'e1': 2, 'eA': 3})
    b = algebra.multivector({'e2': 5, 'eB': 7})
    result = a * b
    assert result.keys() == ('e12', 'e1B', 'e2A', 'eAB')
    assert len(algebra.blades) == algebra.d + 1
