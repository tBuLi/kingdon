from itertools import product, repeat, combinations, chain
from collections import defaultdict
import inspect
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

from sympy import simplify, Symbol
from sympy.utilities.lambdify import lambdify
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
    return NotImplementedError

def codegen_op(x, y):
    """
    Generate the outer product of `x := self` and `y := other`: `x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    res_vals = defaultdict(int)
    for (ei, vi), (ej, vj) in product(x.vals.items(), y.vals.items(), repeat=1):
        if ei ^ ej == ei + ej:
            res_vals[ei ^ ej] += (-1)**x.algebra.swaps[ei, ej] * vi * vj
    # Remove expressions which are identical to zero
    res_vals = {k: simp_expr for k, expr in res_vals.items() if (simp_expr := simplify(expr))}

    return _lambdify(x, y, res_vals)

def codegen_rp(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    raise NotImplementedError

def _lambdify(x, y, vals):
    xy_symbols = list(chain(x.vals.values(), y.vals.values()))
    # TODO: Numba wants a tuple in the line below, but simpy only produces a
    #  list as output if this is a list, not a tuple. See if we can solve this.
    func = lambdify(xy_symbols, list(vals.values()), cse=x.algebra.cse)
    return vals.keys(), njit(func) if x.algebra.numba else func
