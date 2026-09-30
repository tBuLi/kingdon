"""Small exhaustive reference checks for the string blade rules.

The local mask arithmetic is a test oracle for the implementation removed by
issue #140; production code does not use these masks.
"""

from itertools import combinations, product

import pytest

from kingdon import Algebra
from kingdon import blades as rules
import kingdon.operators as ops


def old_swap_blades(blade1, blade2, target=''):
    """Reference swap/count procedure from the removed mask product path."""
    blade1 = list(blade1)
    swaps = 0
    eliminated = []
    for char in blade2:
        if char not in blade1:
            blade1.append(char)
            continue
        idx = blade1.index(char)
        swaps += len(blade1) - idx - 1
        blade1.remove(char)
        eliminated.append(char)
    if target:
        for i, char in enumerate(target):
            idx = blade1.index(char)
            blade1.insert(i, blade1.pop(idx))
            swaps += idx - i
    return swaps, ''.join(blade1), ''.join(eliminated)


def old_oracle(algebra):
    ordered = rules.GENERATOR_LABELS[algebra.start_index:algebra.start_index + algebra.d]
    generators = tuple(blade[1] for blade in algebra.basis if len(blade) == 2) if algebra.basis else ordered
    bit = {generator: 1 << i for i, generator in enumerate(generators)}
    metric = dict(zip(ordered, algebra.signature))
    blades = tuple(algebra.basis) if algebra.basis else tuple(
        blade for g in range(algebra.d + 1)
        for blade in sorted('e' + ''.join(chars) for chars in combinations(ordered, g))
    )
    by_mask = {sum(bit[g] for g in blade[1:]): blade for blade in blades}

    def mask(blade):
        return sum(bit[g] for g in blade[1:])

    def gp(a, b):
        output = by_mask[mask(a) ^ mask(b)]
        swaps, _, eliminated = old_swap_blades(a[1:], b[1:], target=output[1:])
        coefficient = -1 if swaps % 2 else 1
        for generator in eliminated:
            coefficient *= metric[generator]
        return output, coefficient

    def rp(a, b):
        ma, mb = mask(a), mask(b)
        full = (1 << algebra.d) - 1
        output_mask = full - (ma ^ mb)
        output = by_mask[output_mask]
        if full != ma + mb - output_mask:
            return output, 0
        coefficient = (gp(a, by_mask[full - ma])[1]
                       * gp(b, by_mask[full - mb])[1]
                       * gp(by_mask[full - ma], by_mask[full - mb])[1]
                       * gp(by_mask[output_mask], by_mask[ma ^ mb])[1])
        return output, coefficient

    return blades, mask, gp, rp


@pytest.mark.parametrize('algebra', [
    Algebra(4, large=True),
    Algebra(2, 1, 1, large=True),
    Algebra(signature=[0, -1, 1, 1], large=True),
    Algebra.fromname('2DPGA', large=True),
    Algebra.fromname('3DPGA', large=True),
    Algebra(3, basis=['e', 'e2', 'e1', 'e3', 'e21', 'e13', 'e32', 'e213'], large=True),
])
def test_all_blade_pairs_against_removed_mask_path(algebra):
    blades, mask, old_gp, old_rp = old_oracle(algebra)
    for a, b in product(blades, repeat=2):
        output, sign = old_gp(a, b)
        assert rules.product_blades(algebra, a, b) == (output, sign)
        x, y = algebra.blades[a], algebra.blades[b]
        expected = {} if sign == 0 else {output: sign}
        assert dict(ops.gp(x, y).items()) == expected
        reverse_sign = old_gp(b, a)[1]
        assert dict(ops.cp(x, y).items()) == (expected if sign != reverse_sign else {})
        assert dict(ops.acp(x, y).items()) == (expected if sign == reverse_sign else {})

        ma, mb = mask(a), mask(b)
        grade_a, grade_b, grade_out = (len(blade) - 1 for blade in (a, b, output))
        expected_op = expected if ma & mb == 0 else {}
        assert dict(ops.op(x, y).items()) == expected_op
        for operation, accepted in (
            (ops.ip, grade_out == abs(grade_a - grade_b)),
            (ops.lc, grade_out == grade_b - grade_a),
            (ops.rc, grade_out == grade_a - grade_b),
            (ops.sp, grade_out == 0),
        ):
            assert dict(operation(x, y).items()) == (expected if accepted else {})

        rp_output, rp_sign = old_rp(a, b)
        assert dict(ops.rp(x, y).items()) == ({} if not rp_sign else {rp_output: rp_sign})

    full = (1 << algebra.d) - 1
    by_mask = {mask(blade): blade for blade in blades}
    for blade in blades:
        dual = by_mask[full - mask(blade)]
        assert dict(ops.hodge(algebra.blades[blade]).items()) == {dual: old_gp(blade, dual)[1]}
        assert dict(ops.unhodge(algebra.blades[blade]).items()) == {dual: old_gp(dual, blade)[1]}


def test_sparse_large_does_not_materialize_full_basis():
    algebra = Algebra(20, large=True)
    assert not hasattr(algebra, 'blade2mask')
    assert not hasattr(algebra, 'mask2blade')
    assert len(algebra.blades) == algebra.d + 1  # basis vectors and pseudoscalar
    a = algebra.multivector({'e1': 2, 'eA': 3})
    b = algebra.multivector({'e2': 5, 'eB': 7})
    result = a * b
    assert result.keys() == ('e12', 'e1B', 'e2A', 'eAB')
    assert len(algebra.blades) == algebra.d + 1


def test_default_basis_position_and_full_orders():
    for d in (3, 12):
        algebra = Algebra(d, large=True)
        blades = tuple(algebra.indices_for_grades(tuple(range(d + 1))))
        assert all(rules.basis_position(algebra, blade) == i for i, blade in enumerate(blades))
    high = Algebra(27, large=True)
    assert high._generators[-3:] == ('P', 'R', 'Q')
    assert all(rules.basis_position(high, blade) == 1 + high.d + i
               for i, blade in enumerate(high.indices_for_grade(2)))
    algebra = Algebra.fromname('3DPGA', large=True)
    expected_subset_order = (
        'e', 'e1', 'e2', 'e12', 'e3', 'e31', 'e23', 'e123',
        'e0', 'e01', 'e02', 'e021', 'e03', 'e013', 'e032', 'e0123',
    )
    assert tuple(rules.subset_order(algebra)) == expected_subset_order
    assert algebra.blades.e.asfullmv(canonical=False).keys() == expected_subset_order


def test_binary_blade_compatibility_is_removed():
    algebra = Algebra(3)
    for name in ('blade2mask', 'mask2blade', 'canon2bin', 'bin2canon', 'signs'):
        assert not hasattr(algebra, name)
    with pytest.raises(KeyError):
        algebra.multivector({1: 2})
    with pytest.raises(KeyError):
        algebra.multivector(keys=(1,), values=[2])
    with pytest.raises(TypeError):
        type(algebra.blades.e).fromkeysvalues(algebra, (1,), [2])


@pytest.mark.parametrize('basis, message', [
    (['e', 'e1', 'e2'], 'every blade'),
    (['e', 'e1', 'e12', 'e21'], 'every generator blade'),
    (['e', 'e1', 'e2', 'e1'], 'Duplicate generator support'),
    (['e', 'e1', 'e2', 'e11'], 'Invalid custom basis blade'),
    (['e', 'e1', 'e2', 'e1@'], 'Invalid custom basis blade'),
    (['e', 'e12', 'e1', 'e2'], 'grade order'),
])
def test_invalid_custom_basis(basis, message):
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


def test_exact_default_layout_position_outputs():
    algebra = Algebra(3)
    mv = algebra.multivector(keys=('e', 'e3', 'e13', 'e123'), values=[1, 2, 3, 4])
    assert mv.type_number == 0b10101001
    assert format(mv, 'keys_binary') == '10101001'


def test_custom_basis_with_letter_generators():
    algebra = Algebra(2, basis=['e', 'eA', 'eB', 'eBA'])
    assert algebra.start_index == 10
    assert algebra.blades.eB * algebra.blades.eA == algebra.blades.eBA
    assert algebra.blades.eA * algebra.blades.eB == -algebra.blades.eBA


def test_swap_blades():
    """
    Check the removed swap procedure retained as an independent test oracle.
    """
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
        swaps, res_blade, eliminated = old_swap_blades(*test['input'])
        assert swaps == test['output'][0]
        assert res_blade == test['output'][1]
        assert eliminated == test['output'][2]
