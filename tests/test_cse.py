"""
CSE operation count tests, ported from GAmphetamine.test.js.

These tests verify the number of multiplications and additions in the
generated code for common 3DPGA operations. The expected counts match
GAmphetamine's polynomial CSE output, and serve as targets for porting
the polynomial CSE algorithm from polynomial.js to polynomial.py.
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

def get_op_counts(func):
    """Return (muls, adds) from the function docstring, or (None, None)."""
    doc = func.__doc__ or ''
    m = re.match(r'(\d+) muls / (\d+) adds', doc.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


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
    a = pga3d.evenmv(name='a')
    b = pga3d.point(name='b')
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 21
    assert adds == 18

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 84
    assert adds_nc == 33

    # Verify that the two methods produce the same result
    assert c == c_nc

    # Also check that codegen via sympy CSE produces a suboptimal result
    _, func_sp, *_ = do_codegen(codegen_sw, a, b)
    muls_sp, adds_sp = get_op_counts(func_sp)
    assert muls < muls_sp < muls_nc
    assert adds < adds_sp <= adds_nc
    for v1, v2 in zip(func_sp(a.values(), b.values()), c.values()):
        assert not v1 - v2


def test_sw_even_direction(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> direction: CSE 18 muls/12 adds vs no-CSE 60 muls/20 adds."""
    a = pga3d.evenmv(name='a')
    b = pga3d.direction(name='b')
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 18
    assert adds == 12

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.direction(name='b')
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 60
    assert adds_nc == 20

    assert c == c_nc


def test_sw_even_origin(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> origin: CSE 15 muls/9 adds vs no-CSE 24 muls/12 adds."""
    a = pga3d.evenmv(name='a')
    b = pga3d.point()
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 15
    assert adds == 9

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.point()
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 24
    assert adds_nc == 12

    assert c == c_nc


def test_sw_even_e032_half(pga3d, pga3d_no_cse):
    """3DPGA normalized (even >>> e032) / 2: CSE 6 muls/4 adds vs no-CSE 6 muls/4 adds (no improvement)."""
    def _codegen_sw_e032_half(a, b):
        return (a >> b) * 0.5
    
    sw_e032_half = pga3d.compile(symbolic=True)(_codegen_sw_e032_half)
    a = pga3d.evenmv(name='a')
    b = pga3d.direction(e032=1)
    c = sw_e032_half(a, b)
    func_cse = sw_e032_half[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 4

    sw_e032_half_nc = pga3d_no_cse.compile(symbolic=True)(_codegen_sw_e032_half)
    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.direction(e032=1)
    c_nc = sw_e032_half_nc(a_nc, b_nc)
    func_nc = sw_e032_half_nc[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 11
    assert adds_nc == 6

    assert c == c_nc


# ---------------------------------------------------------------------------
# Test 4: geometric product of two even elements
# ---------------------------------------------------------------------------

def test_gp_even_even(pga3d, pga3d_no_cse):
    """3DPGA even * even: CSE 48 muls/40 adds vs no-CSE 48 muls/40 adds (no improvement)."""
    a = pga3d.evenmv(name='a')
    b = pga3d.evenmv(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 48
    assert adds == 40

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.evenmv(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 48
    assert adds_nc == 40

    assert c == c_nc

# ---------------------------------------------------------------------------
# Tests 5-8: geometric products of specialized even elements
# ---------------------------------------------------------------------------

def test_gp_even_translation(pga3d, pga3d_no_cse):
    """3DPGA even * translation: CSE 12 muls/12 adds vs no-CSE 12 muls/12 adds (no improvement)."""
    a = pga3d.evenmv(name='a')
    b = pga3d.translation(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 12
    assert adds == 12

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.translation(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 12
    assert adds_nc == 12

    assert c == c_nc


def test_gp_translation_translation(pga3d, pga3d_no_cse):
    """3DPGA translation * translation: CSE 0 muls/3 adds vs no-CSE 0 muls/3 adds (no improvement)."""
    a = pga3d.translation(name='a')
    b = pga3d.translation(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 0
    assert adds == 3

    a_nc = pga3d_no_cse.translation(name='a')
    b_nc = pga3d_no_cse.translation(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 0
    assert adds_nc == 3

    assert c == c_nc


def test_gp_rotation_rotation(pga3d, pga3d_no_cse):
    """3DPGA rotation * rotation: CSE 9 muls/12 adds vs no-CSE 9 muls/14 adds."""
    a = pga3d.rotation(name='a')
    b = pga3d.rotation(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 12

    a_nc = pga3d_no_cse.rotation(name='a')
    b_nc = pga3d_no_cse.rotation(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert adds_nc == 14

    assert c == c_nc


# ---------------------------------------------------------------------------
# Tests 9-11: regressive product (join)
# ---------------------------------------------------------------------------

def test_rp_point_point(pga3d, pga3d_no_cse):
    """3DPGA join two points: CSE 6 muls/6 adds vs no-CSE 6 muls/10 adds."""
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = a.rp(b)
    func_cse = pga3d.rp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 6

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = a_nc.rp(b_nc)
    func_nc = pga3d_no_cse.rp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 6
    assert adds_nc == 10

    assert c == c_nc


def test_rp_point_line(pga3d, pga3d_no_cse):
    """3DPGA join point and line: CSE 9 muls/9 adds vs no-CSE 9 muls/11 adds."""
    a = pga3d.point(name='a')
    b = pga3d.bivector(name='b')
    c = a.rp(b)
    func_cse = pga3d.rp[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 9

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.bivector(name='b')
    c_nc = a_nc.rp(b_nc)
    func_nc = pga3d_no_cse.rp[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert adds_nc == 11

    assert c == c_nc


def test_rp_three_points(pga3d, pga3d_no_cse):
    """3DPGA join three points: CSE 9 muls/12 adds vs no-CSE 30 muls/22 adds."""
    def _codegen_join3(a, b, c):
        return a.rp(b).rp(c)

    rp_three_points = pga3d.compile(symbolic=True)(_codegen_join3)
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = pga3d.point(name='c')
    d = rp_three_points(a, b, c)
    func_cse = rp_three_points[a, b, c].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 9
    assert adds == 12

    rp_three_points_nc = pga3d_no_cse.compile(symbolic=True)(_codegen_join3)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = pga3d_no_cse.point(name='c')
    d_nc = rp_three_points_nc(a_nc, b_nc, c_nc)
    func_nc = rp_three_points_nc[a_nc, b_nc, c_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 30
    assert adds_nc == 22

    assert d == d_nc


# ---------------------------------------------------------------------------
# Tests 12-14: compound projection expressions
# These require polynomial CSE to match the expected counts.
# ---------------------------------------------------------------------------

def test_project_point_on_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on plane: CSE 18 muls/12 adds vs no-CSE 51 muls/20 adds.

    Expression: (a | b) / b  where a is a normalized point and b is a plane.
    """
    def _codegen_proj_point_plane(a, b):
        return a.ip(b) * b.inv()

    proj_point_plane = pga3d.compile(symbolic=True)(_codegen_proj_point_plane)
    a = pga3d.point(name='a')
    b = pga3d.vector(name='b')
    c = proj_point_plane(a, b)
    func_cse = proj_point_plane[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 18
    assert adds == 12

    proj_point_plane_nc = pga3d_no_cse.compile(symbolic=True)(_codegen_proj_point_plane)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.vector(name='b')
    c_nc = proj_point_plane_nc(a_nc, b_nc)
    func_nc = proj_point_plane_nc[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 51
    assert adds_nc == 20

    assert c == c_nc


def test_project_point_on_normalized_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized plane: CSE 6 muls/6 adds vs no-CSE 24 muls/15 adds.

    Expression: (a | b) * b + (a * (1 - b * ~b)).grade(3)
    This uses the fact that b is normalized (b * ~b = 1), allowing
    polynomial CSE to simplify the second term.
    """
    def _codegen_proj_point_norm_plane(a, b):
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()   # = scalar b*~b
        correction = (a * (1 - b_normsq)).grade(3)
        return ip_ab * b + correction


    proj_point_norm_plane = pga3d.compile(symbolic=True)(_codegen_proj_point_norm_plane)
    a = pga3d.point(name='a')
    b = pga3d.vector(name='b')
    c = proj_point_norm_plane(a, b)
    func_cse = proj_point_norm_plane[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 6
    assert adds == 6

    proj_point_norm_plane_nc = pga3d_no_cse.compile(symbolic=True)(_codegen_proj_point_norm_plane)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.vector(name='b')
    c_nc = proj_point_norm_plane_nc(a_nc, b_nc)
    func_nc = proj_point_norm_plane_nc[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 24
    assert adds_nc == 15

    assert c == c_nc

def test_project_point_on_normalized_line(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized line: CSE 15 muls/15 adds vs no-CSE 39 muls/22 adds.

    Expression: (a | b) * (-b) + (a * (1 - b * ~b)).grade(3)
    """
    def _codegen_proj_point_norm_line(a, b):
        ip_ab = a.ip(b)
        b_normsq = b * b.reverse()
        correction = (a * (1 - b_normsq)).grade(3)
        return ip_ab * (-b) + correction

    proj_point_norm_line = pga3d.compile(symbolic=True)(_codegen_proj_point_norm_line)
    a = pga3d.point(name='a')
    b = pga3d.bivector(name='b')
    c = proj_point_norm_line(a, b)
    func_cse = proj_point_norm_line[a, b].func
    muls, adds = get_op_counts(func_cse)
    assert muls == 15
    assert adds == 15

    proj_point_norm_line_nc = pga3d_no_cse.compile(symbolic=True)(_codegen_proj_point_norm_line)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.bivector(name='b')
    c_nc = proj_point_norm_line_nc(a_nc, b_nc)
    func_nc = proj_point_norm_line_nc[a_nc, b_nc].func
    muls_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 39
    assert adds_nc == 22

    assert c == c_nc
