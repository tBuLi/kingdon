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
    symbolic = alg.multivector(name='x', keys=('e', 'e1', 'e23'))
    numeric = alg.multivector({'e': 1, 'e1': 2, 'e23': 3})
    legacy = alg.multivector(keys=(0, 1, 6), values=[1, 2, 3])
    legacy_mapping = alg.multivector({0: 1, 1: 2, 6: 3})
    legacy_low_level = type(numeric).fromkeysvalues(
        alg, (0, 1, 6), [1, 2, 3]
    )

    assert symbolic.keys() == numeric.keys() == legacy.keys()
    assert legacy_mapping.keys() == legacy_low_level.keys() == legacy.keys()
    assert symbolic.keys() == ('e', 'e1', 'e23')
    assert symbolic.e == Symbol('x')
    assert symbolic.e23 == Symbol('x23')
    assert dict(numeric.items()) == {'e': 1, 'e1': 2, 'e23': 3}
    # Integer membership remains a compatibility input.
    assert 'e23' in numeric and 6 in numeric
    assert_string_keys(
        symbolic, numeric, legacy, legacy_mapping, legacy_low_level
    )

    vector = alg.vector(name='v')
    assert vector.type_layout == {'e1': ..., 'e2': ..., 'e3': ...}
    assert all(
        isinstance(key, str)
        for layout in alg._type_layouts.values()
        for key in layout
    )


def test_vga_products_keep_masks_internal_and_expose_string_keys():
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
    assert list(alg.blade2mask) == [
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

    semantic = alg.multivector(keys=('e31', 'e032'), values=[2, 3])
    legacy = alg.multivector(
        keys=(alg.blade2mask['e31'], alg.blade2mask['e032']),
        values=[2, 3],
    )
    assert semantic.type_number == legacy.type_number

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
