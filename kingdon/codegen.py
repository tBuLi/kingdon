from itertools import product
from collections import defaultdict
from dataclasses import replace
import inspect
from collections import Counter
from functools import partial

from sympy import Symbol, simplify, expand, cse, numbered_symbols
from sympy.utilities.lambdify import lambdify, lambdastr
from numba import njit

def codegen_gp(x, y, symbolic=False):
    """
    Generate the geometric product between `x` and `y`.

    :x: MultiVector
    :y: MultiVector
    :symbolic: If true, return a dict of symbolic expressions instead of lambda functions.
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.vals.items(), y.vals.items(), repeat=1):
        if x.algebra.signs[ei, ej]:
            res_vals[ei ^ ej] += x.algebra.signs[(ei, ej)] * vi * vj
    # Remove expressions which are identical to zero
    res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}
    if symbolic:
        return res_vals

    return _lambdify(x, y, res_vals)

def codegen_sp(x, y):
    """
    Generate the sandwich (conjugation) product between `x` and `y`: `x*y*~x`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    # xyx = x*y*~x
    xy = x.algebra.multivector(vals=codegen_gp(x, y, symbolic=True))
    xyx = codegen_gp(xy, ~x, symbolic=True)
    return _lambdify(x, y, xyx)

def codegen_cp(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    xy = codegen_gp(x, y / 2, symbolic=True)
    yx = codegen_gp(y, x / 2, symbolic=True)
    for k, v in yx.items():
        if k in xy:
            if xy[k] - v:  # Symbolically not equal to zero
                xy[k] -= v
            else:
                del xy[k]
        else:
            xy[k] = - v
    return _lambdify(x, y, xy)

def codegen_ip(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    comm = (x * y - y * x) / 2
    return _lambdify(x, y, comm.vals)

def codegen_op(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    comm = (x * y - y * x) / 2
    return _lambdify(x, y, comm.vals)

def codegen_rp(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    comm = (x * y - y * x) / 2
    return _lambdify(x, y, comm.vals)

def _lambdify(x, y, vals):
    xy_symbols = list(x.vals.values()) + list(y.vals.values())
    # func = lambdify(xy_symbols, tuple(vals.values()), cse=partial(cse, order='none'))
    func = lambdify(xy_symbols, tuple(vals.values()), cse=x.algebra.cse)
    return vals.keys(), njit(func) if x.algebra.numba else func
