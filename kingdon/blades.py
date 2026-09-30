"""String-based rules for oriented basis blades.

Blade labels are the only keys stored in multivectors.  The dictionaries kept for
an explicitly supplied basis describe its orientation and order, not a second
blade representation.
"""

from itertools import combinations, product
from math import comb

GENERATOR_LABELS = '0123456789ABCDEFGHIJKLMNOPRQSTUVWXYZ'


def orientation_parity(source, target):
    positions = {generator: i for i, generator in enumerate(target)}
    indices = [positions[generator] for generator in source]
    return sum(a > b for i, a in enumerate(indices) for b in indices[i + 1:]) & 1


def prepare(algebra):
    generators = tuple(GENERATOR_LABELS[algebra.start_index:algebra.start_index + algebra.d])
    if len(generators) != algebra.d:
        raise ValueError('The algebra has more generators than available blade labels.')
    algebra._generators = generators
    algebra._generator_rank = {g: i for i, g in enumerate(generators)}
    algebra._generator_metric = dict(zip(generators, algebra.signature))
    algebra._subset_order_generators = (
        tuple(blade[1] for blade in algebra.basis if len(blade) == 2)
        if algebra.basis else generators
    )
    algebra._basis_order = {blade: i for i, blade in enumerate(algebra.basis)} if algebra.basis else None
    algebra._input_orientation = {}
    algebra._output_orientation = {}
    if algebra.basis:
        if len(algebra.basis) != 1 << algebra.d or algebra.basis != sorted(algebra.basis, key=len):
            raise ValueError('A custom basis must contain every blade in grade order.')
        if {blade for blade in algebra.basis if len(blade) == 2} != {'e' + g for g in generators}:
            raise ValueError('A custom basis must contain every generator blade.')
        for blade in algebra.basis:
            chars = blade[1:]
            if (not blade.startswith('e') or len(set(chars)) != len(chars)
                    or any(c not in algebra._generator_rank for c in chars)):
                raise ValueError(f'Invalid custom basis blade {blade!r}.')
            default = 'e' + ''.join(sorted(chars, key=algebra._generator_rank.__getitem__))
            if default in algebra._output_orientation:
                raise ValueError(f'Duplicate generator support in custom basis: {blade!r}.')
            parity = orientation_parity(chars, default[1:])
            algebra._input_orientation[blade] = (default, parity)
            algebra._output_orientation[default] = (blade, parity)
        if len(algebra._output_orientation) != 1 << algebra.d:
            raise ValueError('A custom basis must contain every generator support.')


def grade(blade):
    return len(blade) - 1


def commutation_parity(a, b):
    """Whether two nonzero orthogonal blade products anticommute."""
    overlap = sum(generator in b[1:] for generator in a[1:])
    return (grade(a) * grade(b) - overlap) & 1


def normalize(algebra, blade):
    """Return (canonical label, orientation parity), or raise KeyError."""
    if not isinstance(blade, str) or not blade.startswith('e'):
        raise KeyError(blade)
    chars = blade[1:]
    rank = algebra._generator_rank
    if len(set(chars)) != len(chars) or any(c not in rank for c in chars):
        raise KeyError(blade)
    default = 'e' + ''.join(sorted(chars, key=rank.__getitem__))
    parity = orientation_parity(chars, default[1:])
    if algebra.basis:
        canonical, extra = algebra._output_orientation[default]
        return canonical, parity ^ extra
    return default, parity


def is_canonical(algebra, blade):
    if algebra.basis:
        return blade in algebra._input_orientation
    if not isinstance(blade, str) or not blade.startswith('e'):
        return False
    rank = algebra._generator_rank
    try:
        indices = [rank[g] for g in blade[1:]]
    except KeyError:
        return False
    return indices == sorted(set(indices))


def product_blades(algebra, a, b):
    """Geometric product of canonical blades as (canonical string, -1/0/+1)."""
    parity = 0
    if algebra.basis:
        a, pa = algebra._input_orientation[a]
        b, pb = algebra._input_orientation[b]
        parity = pa ^ pb
    left, right = a[1:], b[1:]
    rank, metric = algebra._generator_rank, algebra._generator_metric
    i = j = 0
    n, m = len(left), len(right)
    output = []
    coefficient = 1
    while i < n and j < m:
        x, y = left[i], right[j]
        rx, ry = rank[x], rank[y]
        if rx < ry:
            output.append(x)
            i += 1
        elif rx > ry:
            output.append(y)
            parity ^= (n - i) & 1
            j += 1
        else:
            parity ^= (n - i - 1) & 1
            coefficient *= metric[x]
            i += 1
            j += 1
    if i < n:
        output.append(left[i:])
    if j < m:
        output.append(right[j:])
    default = 'e' + ''.join(output)
    if algebra.basis:
        canonical, extra = algebra._output_orientation[default]
        parity ^= extra
    else:
        canonical = default
    return canonical, -coefficient if parity else coefficient


def complement(algebra, blade):
    """Canonical blade containing exactly the generators absent from blade."""
    present = set(blade[1:])
    default = 'e' + ''.join(g for g in algebra._generators if g not in present)
    return algebra._output_orientation[default][0] if algebra.basis else default


def hodge_blade(algebra, blade, undual=False):
    other = complement(algebra, blade)
    _, sign = product_blades(algebra, other, blade) if undual else product_blades(algebra, blade, other)
    return other, sign


def order_key(algebra, blade):
    return algebra._basis_order[blade] if algebra.basis else (len(blade), blade)


def iter_blades(algebra, grades=None):
    """Explicitly enumerate requested basis grades in observable basis order."""
    grades = set(range(algebra.d + 1) if grades is None else grades)
    if algebra.basis:
        yield from (blade for blade in algebra.basis if grade(blade) in grades)
    else:
        for g in sorted(grades):
            if 0 <= g <= algebra.d:
                labels = ('e' + ''.join(chars) for chars in combinations(algebra._generators, g))
                yield from sorted(labels)


def subset_order(algebra):
    """Explicit full enumeration in subset-counter order."""
    generators = algebra._subset_order_generators
    for flags in product((False, True), repeat=algebra.d):
        selected = {g for g, include in zip(reversed(generators), flags) if include}
        default = 'e' + ''.join(g for g in algebra._generators if g in selected)
        yield algebra._output_orientation[default][0] if algebra.basis else default


def basis_position(algebra, blade):
    """Position in the full observable basis, without enumerating that basis."""
    if algebra.basis:
        return algebra._basis_order[blade]
    chars = blade[1:]
    n, g = algebra.d, len(chars)
    position = sum(comb(n, k) for k in range(g))
    previous = -1
    for slot, target in enumerate(chars):
        target_index = algebra._generator_rank[target]
        remaining = g - slot - 1
        for i in range(previous + 1, n):
            if algebra._generators[i] < target and n - i - 1 >= remaining:
                position += comb(n - i - 1, remaining)
        previous = target_index
    return position
