from itertools import product, combinations, groupby
from collections import namedtuple
from functools import reduce
import operator
import linecache

from sympy import simplify, sympify, Add, Mul
from sympy.utilities.lambdify import lambdify
from numba import njit


TermTuple = namedtuple('TermTuple', ['key_out', 'keys_in', 'sign', 'values_in', 'termstr'])
TermTuple.__doc__ = """
TermTuple represents a single monomial in a product of multivectors.

:param key_out: is the basis blade to which this monomial belongs.
:param keys_in: are the input basis blades in this monomial.
:param sign: Sign of the monomial.
:param values_in: Input values. Typically sympy symbols.
:param termstr: The string representation of this monomial, e.g. '-x*y'.
"""


def term_tuple(items, sign_func, keyout_func=operator.xor):
    """
    Create a single term in a multivector product between the basis blades present in `items`.
    """
    keys_in, values_in = zip(*items)
    sign = reduce(operator.mul, (sign_func(pair) for pair in combinations(keys_in, r=2)))

    if not sign:
        return TermTuple(key_out=0, keys_in=keys_in, sign=sign, values_in=values_in, termstr='')
    # Values could also have signs, e.g. -x1. Multiply signs by these.
    sign_mod, values_in = zip(*(v.args if isinstance(v, Mul) else (1, v) for v in values_in))
    sign = reduce(operator.mul, sign_mod, sign)
    key_out = reduce(keyout_func, keys_in)
    return TermTuple(key_out=key_out,
                     keys_in=keys_in,
                     sign=sign,
                     values_in=values_in,
                     termstr=f'{"+" if sign > 0 else "-"}{"*".join(v.name for v in values_in)}')


def codegen_product(*mvs, name_base, filter_func=lambda tt: tt.sign, sign_func=None, keyout_func=operator.xor, asdict=False, sympy=False):
    """
    Helper function for the codegen of all product-type functions.

    :param *mvs: Positional-argument :class:`~kingdon.algebra.MultiVector`
    :param filter_func: A condition which should be true in the preprocessing of terms.
        For example, for the geometric product we filter out all values for which
        ei*ej = 0 since these do not have to be considered anyway.
        Input is a TermTuple.
    :param sign_func: function to compute sign between terms. E.g. algebra.signs[ei, ej]
        for metric dependent products. Input: 2-tuple of blade indices, e.g. (ei, ej).
    :param name_base: base name for the generated code.
    :param asdict: If true, return the dict of strings before converting to a function.
    """
    sortfunc = lambda x: x.key_out
    algebra = mvs[0].algebra
    if sign_func is None:
        sign_func = lambda pair: algebra.signs[pair]

    terms = filter(filter_func, (term_tuple(items, sign_func, keyout_func=keyout_func)
                                 for items in product(*(mv.items() for mv in mvs))))
    # TODO: Can we loop over the basis blades in such a way that no sort is needed?
    sorted_terms = sorted(terms, key=sortfunc)
    if not sympy:
        res = {k: "".join(term.termstr for term in group)
               for k, group in groupby(sorted_terms, key=sortfunc)}
    else:
        res = {k: Add(*(Mul(*(term.values_in if term.sign == 1 else (term.sign, *term.values_in)), evaluate=False)
                        for term in group), evaluate=False)
               for k, group in groupby(sorted_terms, key=sortfunc)}

    if asdict:
        return res
    elif not sympy:
        return _func_builder(res, *mvs, name_base=name_base)
    else:
        return _lambdify_binary(*mvs, res)


def codegen_gp(x, y, symbolic=False):
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.algebra.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.algebra.MultiVector`.
    :param symbolic: If true, return a symbolic :class:`~kingdon.algebra.MultiVector` instead of a lambda function.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    keys_out, func = codegen_product(x, y, name_base='gp')
    if symbolic:
        return x.fromkeysvalues(x.algebra, keys_out, func(x.values(), y.values()))

    return keys_out, func


def codegen_conj(x, y):
    if x.algebra.simplify:
        res = codegen_product(x, y, ~x, name_base='gp', asdict=True, sympy=True)
        res = {k: str(simp_expr) for k, expr in res.items() if (simp_expr := simplify(expr))}
    else:
        res = codegen_product(x, y, ~x, name_base='gp', asdict=True)
    return _func_builder(res, x, y, name_base="conj")


def codegen_cp(x, y, symbolic=False):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] - algebra.signs[tt.keys_in[::-1]])
    keys_out, func = codegen_product(x, y, filter_func=filter_func, name_base='cp')
    if symbolic:
        return x.fromkeysvalues(algebra, keys_out, func(x.values(), y.values()))

    return keys_out, func


def codegen_acp(x, y, symbolic=False):
    """
    Generate the anti-commutator product of `x := self` and `y := other`: `x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] + algebra.signs[tt.keys_in[::-1]])
    keys_out, func = codegen_product(x, y, filter_func=filter_func, name_base='acp')
    if symbolic:
        return x.fromkeysvalues(algebra, keys_out, func(x.values(), y.values()))

    return keys_out, func


def codegen_ip(x, y, diff_func=abs, symbolic=False):
    """
    Generate the inner product of `x := self` and `y := other`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: tt.key_out == diff_func(tt.keys_in[0] - tt.keys_in[1])
    keys_out, func = codegen_product(x, y, filter_func=filter_func, name_base='ip')
    if symbolic:
        return x.fromkeysvalues(algebra, keys_out, func(x.values(), y.values()))

    return keys_out, func


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
    filter_func = lambda tt: tt.key_out == abs(tt.keys_in[0] - tt.keys_in[1]) ^ tt.keys_in[2]
    res = codegen_product(x, y, ~y, name_base='gp', filter_func=filter_func, asdict=True)
    return _func_builder(res, x, y, name_base='proj')


def codegen_op(x, y, symbolic=False):
    """
    Generate the outer product of `x := self` and `y := other`: `x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: tt.key_out == sum(tt.keys_in)
    sign_func = lambda pair: (-1)**algebra.swaps[pair]
    keys_out, func = codegen_product(x, y, filter_func=filter_func, sign_func=sign_func, name_base='op')
    if symbolic:
        return x.fromkeysvalues(algebra, keys_out, func(x.values(), y.values()))

    return keys_out, func


def codegen_rp(x, y):
    """
    Generate the commutator product of `x := self` and `y := other`: `x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    x_regr_y = (x.dual().op(y.dual())).undual()
    return _lambdify_binary(x, y, x_regr_y)

def codegen_rp(x, y):
    """
    Generate the regressive product of :code:`x` and :code:`y`: :math:`x \vee y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    keyout_func = lambda tot, key_in: len(algebra) - 1 - (key_in ^ tot)
    filter_func = lambda tt: len(algebra) - 1 == sum(tt.keys_in) - tt.key_out
    # Sign is composed of dualization of each blade, exterior product, and undual.
    sign_func = lambda pair: (
        algebra.signs[pair[0], len(algebra) - 1 - pair[0]] *
        algebra.signs[pair[1], len(algebra) - 1 - pair[1]] *
        (-1)**algebra.swaps[len(algebra) - 1 - pair[0], len(algebra) - 1 - pair[1]] *
        algebra.signs[len(algebra) - 1 - (pair[0] ^ pair[1]), pair[0] ^ pair[1]]
    )

    return codegen_product(
        x, y,
        filter_func=filter_func,
        keyout_func=keyout_func,
        sign_func=sign_func,
        name_base='rp'
    )


Fraction = namedtuple('Fraction', ['numer', 'denom'])
Fraction.__doc__ = """
Tuple representing a fraction.
"""

def codegen_inv(x, symbolic=False):
    """
    Generate code for the inverse of :code:`x`.
    Currently, this always uses the Shirokov inverse, which is works in any algebra,
    but it can be expensive to compute.
    In the future this should be extended to use dedicated solutions for known cases.
    """
    k = 2 ** ((x.algebra.d + 1) // 2)
    x_i = x
    if x.grades == (0,):
        adj_x = 1
    else:
        for i in range(1, k + 1):
            # Sympify ratio to keep the ratios exact and avoid floating point errors.
            c_i = (sympify(k) / i) * x_i[0] if x_i[0] else x_i[0]
            adj_x = (x_i - c_i)
            if x.algebra.simplify:
                keys, values = zip(*((k, simp_expr) for k, expr in adj_x.items() if (simp_expr := simplify(expr))))
                adj_x = adj_x.fromkeysvalues(adj_x.algebra, keys=keys, values=values)
            x_i = x * adj_x
            if x_i:
                if x.algebra.simplify:
                    keys, values = zip(*((k, simp_expr) for k, expr in x_i.items() if (simp_expr := simplify(expr))))
                    x_i = x_i.fromkeysvalues(x_i.algebra, keys=keys, values=values)

                if x_i.grades == (0,):
                    break
            else:
                break

    if symbolic:
        return Fraction(adj_x, x_i[0])
    xinv = x.algebra.multivector({k: v / x_i[0] for k, v in adj_x.items()})
    return _lambdify_unary(x, xinv)


def codegen_div(x, y):
    """
    Generate code for :math:`x y^{-1}`.
    """
    adjy, normsqy = codegen_inv(y, symbolic=True)
    if not normsqy:
        raise ZeroDivisionError
    x_adjy = x * adjy
    if x.algebra.simplify:
        xdivy = x.algebra.multivector({k: simplify(v / normsqy) for k, v in x_adjy.items()})
    else:
        xdivy = x.algebra.multivector({k: v / normsqy for k, v in x_adjy.items()})
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

def _func_builder(res_vals: dict, *mvs, name_base: str):
    """
    Build a Python functions for the product between given multivectors.

    :param res_vals: Dict to be converted into a function. The keys correspond to the basis blades in binary,
        while the values are strings to be converted into source code.
    :param mvs: all the multivectors that the resulting function is a product of.
    :param name_base: Base for the name of the function. E.g. 'gp'.
    :return: tuple of output keys of the callable, and the callable.
    """
    args = [f'arg{i}' for i in range(1, len(mvs) + 1)]
    keys_str_per_arg = ["_".join(str(k) for k in mv.keys()) for mv in mvs]
    func_name = f'{name_base}_{"_x_".join(keys_str_per_arg)}'
    header = f'def {func_name}({", ".join(args)}):'
    if res_vals:
        body = "\n".join(f'    {",".join(v.name for v in mv.values())}, = {arg}' for mv, arg in zip(mvs, args))
        return_val = f'    return ({", ".join(res_vals.values())},)'
    else:
        body = ''
        return_val = f'    return tuple()'
    func_source = f'{header}\n{body}\n{return_val}'

    # Dynamically build a function
    func_locals = {}
    c = compile(func_source, func_name, 'exec')
    exec(c, {}, func_locals)

    # Add the generated code to linecache such that it is inspect-safe.
    linecache.cache[func_name] = (len(func_source), None, func_source.splitlines(True), func_name)
    func = func_locals[func_name]
    return tuple(res_vals.keys()), njit(func) if mvs[0].algebra.numba else func
