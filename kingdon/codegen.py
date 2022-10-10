from itertools import product
from collections import defaultdict

from sympy import simplify
from sympy.utilities.lambdify import lambdify
from numba import njit

def codegen_gp(x, y, symbolic=False):
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.algebra.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.algebra.MultiVector`.
    :param symbolic: If true, return a dict of symbolic expressions instead of lambda functions.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.items(), y.items()):
        if x.algebra.signs[ei, ej]:
            res_vals[ei ^ ej] += x.algebra.signs[(ei, ej)] * vi * vj
    # Remove expressions which are identical to zero
    if x.algebra.simplify:
        res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}
    if symbolic:
        return x.fromkeysvalues(x.algebra, tuple(res_vals.keys()), tuple(res_vals.values()))

    return _lambdify_binary(x, y, res_vals)

def codegen_conj(x, y, symbolic=False):
    """
    Generate the sandwich (conjugation) product between :code:`x` and :code:`y`: :code:`x*y*~x`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    xy = codegen_gp(x, y, symbolic=True)
    xyx = codegen_gp(xy, ~x, symbolic=True)
    if symbolic:
        return xyx

    return _lambdify_binary(x, y, xyx)

def codegen_cp(x, y, symbolic=False):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.items(), y.items()):
        if x.algebra.signs[ei, ej] and (x.algebra.signs[(ei, ej)] - x.algebra.signs[(ej, ei)]):
            res_vals[ei ^ ej] += x.algebra.signs[(ei, ej)] * vi * vj

    # Remove expressions which are identical to zero
    if x.algebra.simplify:
        res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}

    if symbolic:
        return x.fromkeysvalues(x.algebra, tuple(res_vals.keys()), tuple(res_vals.values()))

    return _lambdify_binary(x, y, res_vals)


def codegen_acp(x, y, symbolic=False):
    """
    Generate the anti-commutator product of `x := self` and `y := other`: `x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.items(), y.items()):
        if x.algebra.signs[ei, ej] and (x.algebra.signs[(ei, ej)] + x.algebra.signs[(ej, ei)]):
            res_vals[ei ^ ej] += x.algebra.signs[(ei, ej)] * vi * vj

    # Remove expressions which are identical to zero
    if x.algebra.simplify:
        res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}

    if symbolic:
        return x.fromkeysvalues(x.algebra, tuple(res_vals.keys()), tuple(res_vals.values()))

    return _lambdify_binary(x, y, res_vals)


def codegen_ip(x, y, diff_func=abs, symbolic=False):
    """
    Generate the inner product of `x := self` and `y := other`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.items(), y.items()):
        if ei ^ ej == diff_func(ei - ej):
            res_vals[ei ^ ej] += x.algebra.signs[ei, ej] * vi * vj
    if x.algebra.simplify:
        # Remove expressions which are identical to zero
        res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}
    if symbolic:
        return x.fromkeysvalues(x.algebra, tuple(res_vals.keys()), tuple(res_vals.values()))

    return _lambdify_binary(x, y, res_vals)

def codegen_lc(x, y):
    """
    Generate the left-contraction of `x := self` and `y := other`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: -x)

def codegen_rc(x, y):
    """
    Generate the right-contraction of `x := self` and `y := other`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: x)

def codegen_sp(x, y):
    """
    Generate the scalar product of `x := self` and `y := other`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: 0)

def codegen_proj(x, y):
    """
    Generate the projection of `x := self` onto `y := other`: :math:`(x \cdot y) / y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    x_dot_y = codegen_ip(x, y, symbolic=True)
    x_proj_y = codegen_gp(x_dot_y, ~y, symbolic=True)
    return _lambdify_binary(x, y, x_proj_y)

def codegen_op(x, y, symbolic=False):
    """
    Generate the outer product of `x := self` and `y := other`: `x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.items(), y.items()):
        if ei ^ ej == ei + ej:
            res_vals[ei ^ ej] += (-1)**x.algebra.swaps[ei, ej] * vi * vj
    if x.algebra.simplify:
        # Remove expressions which are identical to zero
        res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}
    if symbolic:
        return x.fromkeysvalues(x.algebra, tuple(res_vals.keys()), tuple(res_vals.values()))

    return _lambdify_binary(x, y, res_vals)

def codegen_rp(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    x_regr_y = codegen_op(x.dual(), y.dual(), symbolic=True).undual()
    return _lambdify_binary(x, y, x_regr_y)


def codegen_inv(x, symbolic=False):
    """
    Generate code for the inverse of :code:`x`.
    Currently, this always uses the Shirokov inverse, which is works in any algebra,
    but it can be expensive to compute.
    In the future this should be extended to use dedicated solutions for known cases.
    """
    k = 2 ** ((x.algebra.d + 1) // 2)
    x_i = x
    i = 1
    while x_i.grades != (0,) and x_i:
        c_i = k * x_i[0] / i
        adj_x = (x_i - c_i)
        x_i = codegen_gp(x, adj_x, symbolic=True)
        i += 1
    xinv = adj_x / x_i[0]
    if x.algebra.simplify:
        xinv = x.algebra.multivector({k: simp_expr for k, v in xinv.vals.items() if (simp_expr := simplify(v))})

    if symbolic:
        return xinv

    return _lambdify_unary(x, xinv)


def codegen_div(x, y):
    """
    Generate code for :math:`x y^{-1}`.
    """
    yinv = codegen_inv(y, symbolic=True)
    xdivy = codegen_gp(x, yinv, symbolic=True)
    return _lambdify_binary(x, y, xdivy)


def _lambdify_binary(x, y, x_bin_y):
    xy_symbols = [list(x.values()), list(y.values())]
    func = lambdify(xy_symbols, list(x_bin_y.values()), cse=x.algebra.cse)
    return tuple(x_bin_y.keys()), njit(func) if x.algebra.numba else func


def _lambdify_unary(x, x_unary):
    func = lambdify([list(x.values())], list(x_unary.values()), cse=x.algebra.cse)
    return tuple(x_unary.keys()), njit(func) if x.algebra.numba else func


def _lambdify_mv(free_symbols, mv):
    # TODO: Numba wants a tuple in the line below, but simpy only produces a
    #  list as output if this is a list, not a tuple. See if we can solve this.
    func = lambdify(free_symbols, list(mv.values()), cse=mv.algebra.cse)
    return tuple(mv.keys()), njit(func) if mv.algebra.numba else func
