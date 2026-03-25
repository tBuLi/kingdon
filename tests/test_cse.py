"""
CSE operation count tests, ported from GAmphetamine.test.js.

These tests verify the number of multiplications and additions in the
generated code for common 3DPGA operations. The expected counts match
GAmphetamine's polynomial CSE output, and serve as targets for porting
the polynomial CSE algorithm from polynomial.js to polynomial.py.

Most tests will fail until polynomial CSE is implemented.

# TODO: switch these tests to use highlevel operations like >> and * instead of manually using rational polynomials.
#  However, for now that is ok because strictly speaking these tests are testing the CSE algorithm itself, not the highlevel operations.
"""
import math
import re
import pytest
from kingdon import Algebra, MultiVector
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import do_codegen, codegen_sw, codegen_gp, codegen_rp, codegen_ip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def one():
    return RationalPolynomial([[1]])


def rp(name):
    return RationalPolynomial.fromname(name)


def get_op_counts(func):
    """Return (muls, adds) from the function docstring, or (None, None)."""
    doc = func.__doc__ or ''
    m = re.match(r'(\d+) muls / (\d+) adds', doc.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def check_same_result(func_cse, func_nc, *args):
    """Verify both functions give the same numerical result."""
    r_cse = func_cse(*args)
    r_nc = func_nc(*args)
    assert len(r_cse) == len(r_nc)
    for v_cse, v_nc in zip(r_cse, r_nc):
        assert math.isclose(v_cse, v_nc, rel_tol=1e-9, abs_tol=1e-12), \
            f"CSE={v_cse!r} vs no-CSE={v_nc!r}"


def make_even(alg, prefix='a'):
    """Full even multivector (grades 0,2,4) with symbolic RP coefficients."""
    return alg.evenmv(name=prefix, symbolcls=RationalPolynomial.fromname)


def make_normalized_point(alg, prefix='b'):
    """Trivector with symbolic x,y,z and w=1 (normalized point)."""
    e123 = alg.canon2bin['e123']
    keys = tuple(sorted(alg.indices_for_grade(3)))
    vals = [one() if k == e123 else rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_direction(alg, prefix='b'):
    """Trivector with symbolic x,y,z only (direction, no homogeneous component)."""
    e123 = alg.canon2bin['e123']
    keys = tuple(sorted(k for k in alg.indices_for_grade(3) if k != e123))
    vals = [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in keys]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_origin(alg):
    """Trivector with only e123=1 (the origin point)."""
    e123 = alg.canon2bin['e123']
    return MultiVector.fromkeysvalues(alg, (e123,), [one()])


def make_pure_e032(alg):
    """Single e032 blade with value 1."""
    e032 = alg.canon2bin['e032']
    return MultiVector.fromkeysvalues(alg, (e032,), [one()])


def make_rotation(alg, prefix='a'):
    """Rotation rotor: scalar=1, e12=x, e31=y, e23=z."""
    e_s = alg.canon2bin['e']
    e12 = alg.canon2bin['e12']
    e31 = alg.canon2bin['e31']
    e23 = alg.canon2bin['e23']
    keys = (e_s, e12, e31, e23)
    vals = [one()] + [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in (e12, e31, e23)]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_translation(alg, prefix='a'):
    """Translation motor: scalar=1, e01=x, e02=y, e03=z."""
    e_s = alg.canon2bin['e']
    e01 = alg.canon2bin['e01']
    e02 = alg.canon2bin['e02']
    e03 = alg.canon2bin['e03']
    keys = (e_s, e01, e02, e03)
    vals = [one()] + [rp(f'{prefix}{alg.bin2canon[k][1:]}') for k in (e01, e02, e03)]
    return MultiVector.fromkeysvalues(alg, keys, vals)


def make_vector(alg, prefix='b'):
    """Grade-1 vector (plane in 3DPGA) with all symbolic RP coefficients."""
    return alg.vector(name=prefix, symbolcls=RationalPolynomial.fromname)


def make_bivector(alg, prefix='b'):
    """Grade-2 bivector (line in 3DPGA) with all symbolic RP coefficients."""
    return alg.bivector(name=prefix, symbolcls=RationalPolynomial.fromname)


# ---------------------------------------------------------------------------
# Numerical values for correctness verification
# ---------------------------------------------------------------------------

_NUM_EVEN   = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0]
_NUM_EVEN_2 = [23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0]
_NUM_POINT   = [1.0, 2.0, 3.0, 5.0]    # normalized: e123=1, then e021, e013, e032
_NUM_POINT_2 = [1.0, 7.0, 11.0, 13.0]
_NUM_POINT_3 = [1.0, 17.0, 19.0, 23.0]
_NUM_DIR     = [2.0, 3.0, 5.0]
_NUM_ORIGIN  = [1.0]
_NUM_ROT     = [1.0, 2.0, 3.0, 5.0]    # scalar=1, e12, e31, e23
_NUM_ROT_2   = [1.0, 7.0, 11.0, 13.0]
_NUM_TRANS   = [1.0, 2.0, 3.0, 5.0]    # scalar=1, e01, e02, e03
_NUM_TRANS_2 = [1.0, 7.0, 11.0, 13.0]
_NUM_VEC     = [2.0, 3.0, 5.0, 7.0]    # e1, e2, e3, e0
_NUM_BIV     = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pga3d():
    return Algebra.fromname('3DPGA', cse=True)


@pytest.fixture(scope='module')
def pga3d_no_cse():
    return Algebra.fromname('3DPGA', cse=False)


# ---------------------------------------------------------------------------
# Tests 1-4: sandwich product of even element with various trivectors
# ---------------------------------------------------------------------------

def test_sw_even_normalized_point(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> normalized point: CSE 21 muls/18 adds vs no-CSE 84 muls/33 adds."""
    a = make_even(pga3d)
    b = make_normalized_point(pga3d)
    _, func_cse = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 21
    assert adds == 18

    a_nc = make_even(pga3d_no_cse)
    b_nc = make_normalized_point(pga3d_no_cse)
    _, func_nc = do_codegen(codegen_sw, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 84
    assert adds_nc == 33

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_POINT)

    # Also check that codegen via sympy produces a suboptimal result
    a_sp = a.map(lambda v: v.tosympy())
    b_sp = b.map(lambda v: v.tosympy())
    _, func_sp = do_codegen(codegen_sw, a_sp, b_sp)
    muls_sp, adds_sp = get_op_counts(func_sp)
    assert muls_sp == 68
    assert adds_sp == 30
    check_same_result(func_sp, func_nc, _NUM_EVEN, _NUM_POINT)


def test_sw_even_direction(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> direction: CSE 18 muls/12 adds vs no-CSE 60 muls/20 adds."""
    a = make_even(pga3d)
    b = make_direction(pga3d)
    _, func_cse = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 18
    assert adds == 12

    a_nc = make_even(pga3d_no_cse)
    b_nc = make_direction(pga3d_no_cse)
    _, func_nc = do_codegen(codegen_sw, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 60
    assert adds_nc == 20

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_DIR)


def test_sw_even_origin(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> origin: CSE 15 muls/9 adds vs no-CSE 24 muls/12 adds."""
    a = make_even(pga3d)
    b = make_origin(pga3d)
    _, func_cse = do_codegen(codegen_sw, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 15
    assert adds == 9

    a_nc = make_even(pga3d_no_cse)
    b_nc = make_origin(pga3d_no_cse)
    _, func_nc = do_codegen(codegen_sw, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 24
    assert adds_nc == 12

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_ORIGIN)


def test_sw_even_e032_half(pga3d, pga3d_no_cse):
    """3DPGA normalized (even >>> e032) / 2: CSE 6 muls/4 adds vs no-CSE 6 muls/4 adds (no improvement)."""
    def codegen_sw_e032_half(a, b):
        return (a >> b) * 0.5
    a = make_even(pga3d)
    b = make_pure_e032(pga3d)
    _, func_cse = do_codegen(codegen_sw_e032_half, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 4

    a_nc = make_even(pga3d_no_cse)
    b_nc = make_pure_e032(pga3d_no_cse)
    _, func_nc = do_codegen(codegen_sw_e032_half, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 11
    assert adds_nc == 6

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_ORIGIN)


# ---------------------------------------------------------------------------
# Test 4: geometric product of two even elements
# ---------------------------------------------------------------------------

def test_gp_even_even(pga3d, pga3d_no_cse):
    """3DPGA even * even: CSE 48 muls/40 adds vs no-CSE 48 muls/40 adds (no improvement)."""
    a = make_even(pga3d, 'a')
    b = make_even(pga3d, 'b')
    _, func_cse = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 48
    assert adds == 40

    a_nc = make_even(pga3d_no_cse, 'a')
    b_nc = make_even(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_gp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 48
    assert adds_nc == 40

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_EVEN_2)

# ---------------------------------------------------------------------------
# Tests 5-8: geometric products of specialized even elements
# ---------------------------------------------------------------------------

def test_gp_even_translation(pga3d, pga3d_no_cse):
    """3DPGA even * translation: CSE 12 muls/12 adds vs no-CSE 12 muls/12 adds (no improvement)."""
    a = make_even(pga3d, 'a')
    b = make_translation(pga3d, 'b')
    _, func_cse = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 12
    assert adds == 12

    a_nc = make_even(pga3d_no_cse, 'a')
    b_nc = make_translation(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_gp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 12
    assert adds_nc == 12

    check_same_result(func_cse, func_nc, _NUM_EVEN, _NUM_TRANS)


def test_gp_translation_translation(pga3d, pga3d_no_cse):
    """3DPGA translation * translation: CSE 0 muls/3 adds vs no-CSE 0 muls/3 adds (no improvement)."""
    a = make_translation(pga3d, 'a')
    b = make_translation(pga3d, 'b')
    _, func_cse = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 0
    assert adds == 3

    a_nc = make_translation(pga3d_no_cse, 'a')
    b_nc = make_translation(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_gp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 0
    assert adds_nc == 3

    check_same_result(func_cse, func_nc, _NUM_TRANS, _NUM_TRANS_2)


def test_gp_rotation_rotation(pga3d, pga3d_no_cse):
    """3DPGA rotation * rotation: CSE 9 muls/12 adds vs no-CSE 9 muls/14 adds."""
    a = make_rotation(pga3d, 'a')
    b = make_rotation(pga3d, 'b')
    _, func_cse = do_codegen(codegen_gp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 12

    a_nc = make_rotation(pga3d_no_cse, 'a')
    b_nc = make_rotation(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_gp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert adds_nc == 14

    check_same_result(func_cse, func_nc, _NUM_ROT, _NUM_ROT_2)


# ---------------------------------------------------------------------------
# Tests 9-11: regressive product (join)
# ---------------------------------------------------------------------------

def test_rp_point_point(pga3d, pga3d_no_cse):
    """3DPGA join two points: CSE 6 muls/6 adds vs no-CSE 6 muls/10 adds."""
    a = make_normalized_point(pga3d, 'a')
    b = make_normalized_point(pga3d, 'b')
    _, func_cse = do_codegen(codegen_rp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 6

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_normalized_point(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_rp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 6
    assert adds_nc == 10

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_POINT_2)


def test_rp_point_line(pga3d, pga3d_no_cse):
    """3DPGA join point and line: CSE 9 muls/9 adds vs no-CSE 9 muls/11 adds."""
    a = make_normalized_point(pga3d, 'a')
    b = make_bivector(pga3d, 'b')
    _, func_cse = do_codegen(codegen_rp, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 9

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_bivector(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_rp, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert adds_nc == 11

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_BIV)


def test_rp_three_points(pga3d, pga3d_no_cse):
    """3DPGA join three points: CSE 9 muls/12 adds vs no-CSE 30 muls/22 adds."""
    def codegen_join3(a, b, c):
        return a.rp(b).rp(c)

    a = make_normalized_point(pga3d, 'a')
    b = make_normalized_point(pga3d, 'b')
    c = make_normalized_point(pga3d, 'c')
    _, func_cse = do_codegen(codegen_join3, a, b, c)
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 12

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_normalized_point(pga3d_no_cse, 'b')
    c_nc = make_normalized_point(pga3d_no_cse, 'c')
    _, func_nc = do_codegen(codegen_join3, a_nc, b_nc, c_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 30
    assert adds_nc == 22

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_POINT_2, _NUM_POINT_3)


# ---------------------------------------------------------------------------
# Tests 12-14: compound projection expressions
# These require polynomial CSE to match the expected counts.
# ---------------------------------------------------------------------------

def test_project_point_on_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on plane: CSE 18 muls/12 adds vs no-CSE 51 muls/20 adds.

    Expression: (a | b) / b  where a is a normalized point and b is a plane.
    """
    def codegen_proj_point_plane(a, b):
        return a.ip(b) * b.inv()

    a = make_normalized_point(pga3d, 'a')
    b = make_vector(pga3d, 'b')
    _, func_cse = do_codegen(codegen_proj_point_plane, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 18
    assert adds == 12

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_vector(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_proj_point_plane, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 51
    assert adds_nc == 20

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_VEC)


def test_project_point_on_normalized_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized plane: CSE 6 muls/6 adds vs no-CSE 24 muls/15 adds.

    Expression: (a | b) * b + (a * (1 - b * ~b)).grade(3)
    This uses the fact that b is normalized (b * ~b = 1), allowing
    polynomial CSE to simplify the second term.
    """
    def codegen_proj_point_norm_plane(a, b):
        alg = a.algebra
        e = MultiVector.fromkeysvalues(alg, (0,), [one()])  # scalar 1
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()   # = scalar b*~b
        correction = (a * (e - b_normsq)).grade(3)
        return ip_ab * b + correction

    a = make_normalized_point(pga3d, 'a')
    b = make_vector(pga3d, 'b')
    _, func_cse = do_codegen(codegen_proj_point_norm_plane, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 6

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_vector(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_proj_point_norm_plane, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 24
    assert adds_nc == 15

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_VEC)


def test_project_point_on_normalized_line(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized line: CSE 15 muls/15 adds vs no-CSE 39 muls/22 adds.

    Expression: (a | b) * (-b) + (a * (1 - b * ~b)).grade(3)
    """
    def codegen_proj_point_norm_line(a, b):
        alg = a.algebra
        e = MultiVector.fromkeysvalues(alg, (0,), [one()])
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()
        correction = (a * (e - b_normsq)).grade(3)
        return ip_ab * (-b) + correction

    a = make_normalized_point(pga3d, 'a')
    b = make_bivector(pga3d, 'b')
    _, func_cse = do_codegen(codegen_proj_point_norm_line, a, b)
    muls, adds = get_op_counts(func_cse)
    assert muls == 15
    assert adds == 15

    a_nc = make_normalized_point(pga3d_no_cse, 'a')
    b_nc = make_bivector(pga3d_no_cse, 'b')
    _, func_nc = do_codegen(codegen_proj_point_norm_line, a_nc, b_nc)
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 39
    assert adds_nc == 22

    check_same_result(func_cse, func_nc, _NUM_POINT, _NUM_BIV)
