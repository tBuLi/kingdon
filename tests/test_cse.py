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
from sympy import Symbol
from kingdon import Algebra, MultiVector, Translation
from kingdon.polynomial import RationalPolynomial
from kingdon.codegen import do_compile_symbolic
from kingdon.operators import sw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_op_counts(func):
    """Return (muls, divs, adds) from the function docstring, or (None, None, None)."""
    doc = func.__doc__ or ''
    m = re.search(r'(\d+)\s*muls\s*/\s*(\d+)\s*divs\s*/\s*(\d+)\s*adds', doc)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pga3d():
    return Algebra.fromname('3DPGA', cse=True)


@pytest.fixture(scope='module')
def pga2d():
    return Algebra.fromname('2DPGA', cse=True)


@pytest.fixture(scope='module')
def pga3d_no_cse():
    return Algebra.fromname('3DPGA', cse=False)


@pytest.fixture(scope='module')
def pga2d_no_cse():
    return Algebra.fromname('2DPGA', cse=False)


# ---------------------------------------------------------------------------
# Tests 1-4: sandwich product of even element with various trivectors
# ---------------------------------------------------------------------------

def test_sw_even_normalized_point(pga3d, pga3d_no_cse):
    """
    3DPGA normalized bireflection >> normalized point: CSE 21 muls/18 adds vs no-CSE 72 muls/30 adds.

    GAmphetamine reaches 21/18 with a full 8-component motor (grades 0, 2 and 4) because its type
    carries ``condition: b => 1-b*~b``, the *whole* normalization equation. In 3DPGA that is two
    constraints, ``<b~b>_0 = 1`` and ``<b~b>_4 = 0``, which it turns into polynomial rewrite rules.
    Kingdon's ``ops.sw`` only bakes in the scalar one (the ``.grade(0)`` in its ``condition``), which
    is why our evenmv costs 24/21: three surviving ``a0123*a..`` terms. Measured in GAmphetamine:

        8-component motor, no condition : 28/21     with condition : 21/18
        7-component rotor,  no condition : 25/18     with condition : 18/15

    So without constraints kingdon is ahead (24/21 and 21/18 vs their 28/21 and 25/18), but once the
    constraint system lands the target for this test becomes 18/15, not 21/18. Note that the
    grade-4 part is genuinely needed: no ``a0123``-free polynomial representative exists modulo the
    constraint ideal, and GAmphetamine's 21/18 output does still read ``a[7]``.
    """
    a = pga3d.bireflection(name='a')
    b = pga3d.point(name='b')
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 21
    assert divs == 0
    assert adds == 18

    a_nc = pga3d_no_cse.bireflection(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 72
    assert divs_nc == 0
    assert adds_nc == 30

    # Verify that the two methods produce the same result
    assert not c - c_nc

    # Also check that codegen via sympy CSE produces a suboptimal result
    func_sp = do_compile_symbolic(sw, a, b).func
    muls_sp, divs_sp, adds_sp = get_op_counts(func_sp)
    assert divs_sp == 0
    assert muls < muls_sp < muls_nc
    assert adds < adds_sp
    for v1, v2 in zip(func_sp(a.values(), b.values()), c.values()):
        assert not (v1 - v2).expand()


def test_sw_even_direction(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> direction: CSE 18 muls/12 adds vs no-CSE 60 muls/20 adds."""
    a = pga3d.evenmv(name='a')
    b = pga3d.direction(name='b')
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 18
    assert divs == 0
    assert adds == 12

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.direction(name='b')
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 54
    assert divs_nc == 0
    assert adds_nc == 20

    assert not (c - c_nc)


def test_sw_even_origin(pga3d, pga3d_no_cse):
    """3DPGA normalized even >>> origin: CSE 15 muls/9 adds vs no-CSE 24 muls/12 adds."""
    a = pga3d.evenmv(name='a')
    b = pga3d.point()
    c = a >> b
    func_cse = pga3d.sw[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 15
    assert divs == 0
    assert adds == 9

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.point()
    c_nc = a_nc >> b_nc
    func_nc = pga3d_no_cse.sw[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 24
    assert divs_nc == 0
    assert adds_nc == 12

    assert not c - c_nc


def test_sw_even_e032_half(pga3d, pga3d_no_cse):
    """3DPGA normalized (even >>> e032) / 2: CSE 6 muls/4 adds vs no-CSE 9 muls/6 adds and sympy CSE 6 muls/4 adds."""
    def _codegen_sw_e032_half(a, b):
        return (a >> b) * 0.5

    a = pga3d.evenmv(name='a', symbolcls=RationalPolynomial.fromname)
    b = pga3d.direction(e032=1)
    sw_e032_half = pga3d.compile(_codegen_sw_e032_half, a, b)
    c = sw_e032_half(a, b)
    func_cse = sw_e032_half.func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 6
    assert divs == 0
    assert adds == 4

    a_nc = pga3d_no_cse.evenmv(name='a', symbolcls=RationalPolynomial.fromname)
    b_nc = pga3d_no_cse.direction(e032=1)
    sw_e032_half_nc = pga3d_no_cse.compile(_codegen_sw_e032_half, a_nc, b_nc)
    c_nc = sw_e032_half_nc(a_nc, b_nc)
    func_nc = sw_e032_half_nc.func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert divs_nc == 0
    assert adds_nc == 6

    assert c == c_nc

    # With sympy CSE
    a_sp = pga3d.evenmv(name='a', symbolcls=Symbol)
    b_sp = pga3d.direction(e032=1)
    sw_e032_half_sp = pga3d.compile(_codegen_sw_e032_half, a_sp, b_sp)
    c_sp = sw_e032_half_sp(a_sp, b_sp)
    func_sp = sw_e032_half_sp.func
    muls_sp, divs_sp, adds_sp = get_op_counts(func_sp)
    assert muls_sp == 11
    assert divs_sp == 0
    assert adds_sp == 6

    assert not c.map(lambda v: v.tosympy()) - c_sp


# ---------------------------------------------------------------------------
# Test 4: geometric product of two even elements
# ---------------------------------------------------------------------------

def test_gp_even_even(pga3d, pga3d_no_cse):
    """3DPGA even * even: CSE 48 muls/40 adds vs no-CSE 48 muls/40 adds (no improvement)."""
    a = pga3d.evenmv(name='a')
    b = pga3d.evenmv(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 48
    assert divs == 0
    assert adds == 40

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.evenmv(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 48
    assert divs_nc == 0
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
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 12
    assert divs == 0
    assert adds == 12

    a_nc = pga3d_no_cse.evenmv(name='a')
    b_nc = pga3d_no_cse.translation(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 12
    assert divs_nc == 0
    assert adds_nc == 12

    assert c == c_nc


def test_gp_translation_translation(pga3d, pga3d_no_cse):
    """3DPGA translation * translation: CSE 0 muls/3 adds vs no-CSE 0 muls/3 adds (no improvement)."""
    a = pga3d.translation(name='a')
    b = pga3d.translation(name='b')
    c = a * b
    func_cse = pga3d.gp[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 0
    assert divs == 0
    assert adds == 3

    a_nc = pga3d_no_cse.translation(name='a')
    b_nc = pga3d_no_cse.translation(name='b')
    c_nc = a_nc * b_nc
    func_nc = pga3d_no_cse.gp[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 0
    assert divs_nc == 0
    assert adds_nc == 3

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
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 6
    assert divs == 0
    assert adds == 6

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = a_nc.rp(b_nc)
    func_nc = pga3d_no_cse.rp[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 6
    assert divs_nc == 0
    assert adds_nc == 10

    assert c == c_nc


def test_rp_point_line(pga3d, pga3d_no_cse):
    """3DPGA join point and line: CSE 9 muls/9 adds vs no-CSE 9 muls/11 adds."""
    a = pga3d.point(name='a')
    b = pga3d.bivector(name='b')
    c = a.rp(b)
    func_cse = pga3d.rp[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 9
    assert divs == 0
    assert adds == 9

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.bivector(name='b')
    c_nc = a_nc.rp(b_nc)
    func_nc = pga3d_no_cse.rp[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 9
    assert divs_nc == 0
    assert adds_nc == 11

    assert c == c_nc


def test_rp_three_points(pga3d, pga3d_no_cse):
    """3DPGA join three points: CSE 9 muls/12 adds vs no-CSE 30 muls/22 adds."""
    def _codegen_join3(a, b, c):
        return a.rp(b).rp(c)

    rp_three_points = pga3d.jit(symbolic=True)(_codegen_join3)
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = pga3d.point(name='c')
    d = rp_three_points(a, b, c)
    func_cse = rp_three_points[a, b, c].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 9
    assert divs == 0
    assert adds == 12

    rp_three_points_nc = pga3d_no_cse.jit(symbolic=True)(_codegen_join3)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = pga3d_no_cse.point(name='c')
    d_nc = rp_three_points_nc(a_nc, b_nc, c_nc)
    func_nc = rp_three_points_nc[a_nc, b_nc, c_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 30
    assert divs_nc == 0
    assert adds_nc == 22

    assert not d - d_nc


# ---------------------------------------------------------------------------
# Tests 12-14: compound projection expressions
# These require polynomial CSE to match the expected counts.
# ---------------------------------------------------------------------------

def test_project_point_on_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on plane: CSE 18 muls/12 adds vs no-CSE 51 muls/20 adds.

    Expression: (a | b) / b  where a is a normalized point and b is a plane.
    Does not use a @ b because b is not normalized.
    """
    def _codegen_proj_point_plane(a, b):
        return a.ip(b) * b.inv()

    proj_point_plane = pga3d.jit(symbolic=True)(_codegen_proj_point_plane)
    a = pga3d.point(name='a')
    b = pga3d.vector(name='b')
    c = proj_point_plane(a, b)
    func_cse = proj_point_plane[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 18
    assert divs == 3
    assert adds == 12

    proj_point_plane_nc = pga3d_no_cse.jit(symbolic=True)(_codegen_proj_point_plane)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.vector(name='b')
    c_nc = proj_point_plane_nc(a_nc, b_nc)
    func_nc = proj_point_plane_nc[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 36
    assert divs_nc == 3
    assert adds_nc == 20

    assert not c - c_nc


def test_project_point_on_normalized_plane(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized plane: CSE 6 muls/6 adds vs no-CSE 24 muls/15 adds.

    Expression: ((a | b) * (~b) + (a * (1 - b * ~b))).grade(3)
    This uses the fact that b is normalized (b * ~b = 1), allowing
    polynomial CSE to simplify the second term.
    """
    a = pga3d.point(name='a')
    b = pga3d.vector(name='b')
    c = a @ b
    func_cse = pga3d.proj[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 6
    assert divs == 0
    assert adds == 6

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.vector(name='b')
    c_nc = a_nc @ b_nc
    func_nc = pga3d_no_cse.proj[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 21
    assert divs_nc == 0
    assert adds_nc == 15

    assert not (c - c_nc)

def test_project_point_on_normalized_line(pga3d, pga3d_no_cse):
    """3DPGA project point on normalized line: CSE 12 muls/12 adds vs no-CSE 39 muls/19 adds.

    Expression: ((a | b) * (~b) + (a * (1 - b * ~b))).grade(3)
    """
    a = pga3d.point(name='a')
    b = pga3d.bivector(name='b')
    c = a @ b
    func_cse = pga3d.proj[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 12
    assert divs == 0
    assert adds == 12

    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.bivector(name='b')
    c_nc = a_nc @ b_nc
    func_nc = pga3d_no_cse.proj[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 33
    assert divs_nc == 0
    assert adds_nc == 19

    assert not (c - c_nc)


def test_reflect_point_in_normalized_plane(pga3d, pga3d_no_cse):
    """3DPGA reflect point in normalized plane: CSE 9 muls/7 adds vs no-CSE 33 muls/13 adds."""
    P = pga3d.vector(name='P')
    p = pga3d.point(name='p')
    c = P >> p
    func_cse = pga3d.sw[P, p].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 9
    assert divs == 0
    assert adds == 7

    P_nc = pga3d_no_cse.vector(name='P')
    p_nc = pga3d_no_cse.point(name='p')
    c_nc = P_nc >> p_nc
    func_nc = pga3d_no_cse.sw[P_nc, p_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 33
    assert divs_nc == 0
    assert adds_nc == 13

    assert not (c - c_nc)


def test_reflect_point_in_normalized_line(pga3d, pga3d_no_cse):
    """3DPGA reflect point in normalized line: CSE 15 muls/12 adds vs no-CSE 48 muls/20 adds."""
    l = pga3d.bivector(name='l')
    p = pga3d.point(name='p')
    c = l >> p
    func_cse = pga3d.sw[l, p].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 15
    assert divs == 0
    assert adds == 12

    l_nc = pga3d_no_cse.bivector(name='l')
    p_nc = pga3d_no_cse.point(name='p')
    c_nc = l_nc >> p_nc
    func_nc = pga3d_no_cse.sw[l_nc, p_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 48
    assert divs_nc == 0
    assert adds_nc == 20

    assert not (c - c_nc)


# ---------------------------------------------------------------------------
# Tests 15-17: the k-simplex measures from https://bivector.net/CLEANUP.html
# The CSE targets are the muls/adds of the vector algebra equivalents, since
# the point of that table is that the PGA expressions compile down to them.
# The square root and the scalar prefactor are ignored on both sides, so these
# are the *squared* length and area, and 6x the signed volume.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason='Needs the factoring step of the most recent GAmphetamine polynomial CSE.')
def test_norm_sq_join_two_points(pga3d, pga3d_no_cse):
    """3DPGA squared length of a line segment: CSE 3 muls/5 adds vs no-CSE 12 muls/8 adds.

    Expression: (a & b) * ~(a & b), the PGA equivalent of (b - a) | (b - a).
    The join by itself is 6 muls/6 adds (see test_rp_point_point), but the ideal
    parts do not participate in the norm, so taking the norm *reduces* the count.
    """
    def _codegen_norm_sq_join2(a, b):
        return (a & b) * ~(a & b)

    norm_sq_join2 = pga3d.jit(symbolic=True)(_codegen_norm_sq_join2)
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = norm_sq_join2(a, b)
    func_cse = norm_sq_join2[a, b].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 3
    assert divs == 0
    assert adds == 5

    norm_sq_join2_nc = pga3d_no_cse.jit(symbolic=True)(_codegen_norm_sq_join2)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = norm_sq_join2_nc(a_nc, b_nc)
    func_nc = norm_sq_join2_nc[a_nc, b_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 12
    assert divs_nc == 0
    assert adds_nc == 8

    assert not c - c_nc


@pytest.mark.skip(reason='Needs the factoring step of the most recent GAmphetamine polynomial CSE.')
def test_norm_sq_join_three_points(pga3d, pga3d_no_cse):
    """3DPGA squared area of a triangle: CSE 9 muls/11 adds vs no-CSE 234 muls/62 adds.

    Expression: (a & b & c) * ~(a & b & c), the PGA equivalent of the squared
    norm of the cross product (b - a) x (c - a).
    """
    def _codegen_norm_sq_join3(a, b, c):
        return (a & b & c) * ~(a & b & c)

    norm_sq_join3 = pga3d.jit(symbolic=True)(_codegen_norm_sq_join3)
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = pga3d.point(name='c')
    d = norm_sq_join3(a, b, c)
    func_cse = norm_sq_join3[a, b, c].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 9
    assert divs == 0
    assert adds == 11

    norm_sq_join3_nc = pga3d_no_cse.jit(symbolic=True)(_codegen_norm_sq_join3)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = pga3d_no_cse.point(name='c')
    d_nc = norm_sq_join3_nc(a_nc, b_nc, c_nc)
    func_nc = norm_sq_join3_nc[a_nc, b_nc, c_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 234
    assert divs_nc == 0
    assert adds_nc == 62

    assert not d - d_nc


@pytest.mark.skip(reason='Needs the factoring step of the most recent GAmphetamine polynomial CSE.')
def test_join_four_points(pga3d, pga3d_no_cse):
    """3DPGA signed volume of a tetrahedron: CSE 9 muls/14 adds vs no-CSE 48 muls/24 adds.

    Expression: a & b & c & d, the PGA equivalent of the scalar triple product
    ((b - a) x (c - a)) | (d - a). Like its vector counterpart it is already a
    scalar, so there is no need to square it first.
    """
    def _codegen_join4(a, b, c, d):
        return a & b & c & d

    join4 = pga3d.jit(symbolic=True)(_codegen_join4)
    a = pga3d.point(name='a')
    b = pga3d.point(name='b')
    c = pga3d.point(name='c')
    d = pga3d.point(name='d')
    e = join4(a, b, c, d)
    func_cse = join4[a, b, c, d].func
    muls, divs, adds = get_op_counts(func_cse)
    assert muls == 9
    assert divs == 0
    assert adds == 14

    join4_nc = pga3d_no_cse.jit(symbolic=True)(_codegen_join4)
    a_nc = pga3d_no_cse.point(name='a')
    b_nc = pga3d_no_cse.point(name='b')
    c_nc = pga3d_no_cse.point(name='c')
    d_nc = pga3d_no_cse.point(name='d')
    e_nc = join4_nc(a_nc, b_nc, c_nc, d_nc)
    func_nc = join4_nc[a_nc, b_nc, c_nc, d_nc].func
    muls_nc, divs_nc, adds_nc = get_op_counts(func_nc)
    assert muls_nc == 48
    assert divs_nc == 0
    assert adds_nc == 24

    assert not e - e_nc


def test_rotate_constant_blade():
    """
    Test the compilation for the rotation of a unit vector.
    Because it is a unit vector, all the extra mulls by one can be eliminated.
    """
    alg = Algebra(3)
    alg_nc = Algebra(3, cse=False)
    def rotate_blade(R, blade):
        return R >> blade

    # Rotate a unit vector
    R = alg.bireflection(name='R', symbolcls=alg.codegen_symbolcls)
    e1 = alg.vector(e1=1)
    rotate_e1 = alg.compile(rotate_blade, R, e1)
    e1p = rotate_e1(R, e1)
    muls, divs, adds = get_op_counts(rotate_e1.func)
    assert muls == 9
    assert divs == 0
    assert adds == 5

    # Rotate the non-unit equivalent but still do CSE
    v = alg.vector(e1=1)
    w = R >> v
    muls, divs, adds = get_op_counts(alg.sw[R, v].func)
    assert muls == 14
    assert divs == 0
    assert adds == 5

    assert not e1p - w

    # Rotate the non-unit equivalent but also no CSE
    v_nc = alg_nc.vector(e1=1)
    R_nc = alg.bireflection(name='R', symbolcls=alg.codegen_symbolcls)
    w_nc = R_nc >> v_nc
    muls, divs, adds = get_op_counts(alg_nc.sw[R_nc, v_nc].func)
    assert muls == 18
    assert divs == 0
    assert adds == 7

    assert not e1p - w_nc


def test_inv_div(pga2d, pga2d_no_cse):
    u = pga2d.multivector(name='u', symbolcls=RationalPolynomial.fromname)
    # Multiply by inverse results in a scalar exp, which numerically evaluates to 1.
    def u_uinv(u):
        return u*u.inv()
    func_u_uinv = pga2d.compile(u_uinv, u)
    muls, divs, adds = get_op_counts(func_u_uinv.func)
    assert muls == 0
    assert divs == 0
    assert adds == 0
    assert isinstance(func_u_uinv(u), Translation)
    assert func_u_uinv(u).keys() == ()
    assert func_u_uinv(u).shape == ()

    def udivu(u): return u / u
    func_udivu = pga2d.compile(udivu, u)
    muls, divs, adds = get_op_counts(func_udivu.func)
    assert muls == 0
    assert divs == 0
    assert adds == 0
    assert isinstance(func_udivu(u), Translation)
    assert func_udivu(u).keys() == ()
    assert func_udivu(u).shape == ()

    # Now without CSE. Inversion works too well it seems, even here it is already symbolically 1 before CSE comes into the picture.
    u = pga2d_no_cse.multivector(name='u', symbolcls=RationalPolynomial.fromname)
    func_uinv_nc = pga2d_no_cse.compile(u_uinv, u)
    muls_nc, divs_nc, adds_nc = get_op_counts(func_uinv_nc.func)
    assert muls_nc == 0
    assert divs_nc == 0
    assert adds_nc == 0
    assert isinstance(func_udivu(u), Translation)
    assert func_udivu(u).keys() == ()
    assert func_udivu(u).shape == ()
