from __future__ import annotations

import string
from itertools import product, combinations, groupby, chain
from collections import namedtuple, defaultdict
from typing import NamedTuple, Callable, Tuple, Dict, Optional, List, Type
from functools import reduce, cached_property
import linecache
import warnings
import operator
from dataclasses import dataclass
import inspect
import builtins
import keyword
import copy

from sympy.utilities.iterables import iterable, flatten
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.simplify.cse_main import numbered_symbols
from sympy import Symbol

from kingdon.powers import power_supply
from kingdon.polynomial import poly_cse, poly_format, Polynomial, RationalPolynomial


class CodegenOutput(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    """
    keys_out: Tuple[int]
    func: Callable
    MVType: Type = None


def codegen_product(x, y, filter_func=None, sign_func=None, keyout_func=operator.xor):
    """
    Helper function for the codegen of all product-type functions.

    :param x: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param filter_func: A condition which should be true in the preprocessing of terms.
        Input is a TermTuple.
    :param sign_func: function to compute sign between terms. E.g. algebra.signs[ei, ej]
        for metric dependent products. Input: 2-tuple of blade indices, e.g. (ei, ej).
    :param keyout_func:
    """
    sign_func = sign_func or (lambda pair: x.algebra.signs[pair])

    res = {}
    for (kx, vx), (ky, vy) in product(x.items(), y.items()):
        if (sign := sign_func((kx, ky))):
            key_out = keyout_func(kx, ky)
            if filter_func and not filter_func(kx, ky, key_out): continue
            termstr = vx * vy if sign > 0 else (- vx * vy)
            if key_out in res:
                res[key_out] += termstr
            else:
                res[key_out] = termstr
    return res


def codegen_gp(x, y):
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    return codegen_product(x, y)


def codegen_sw(x, y):
    r"""
    Generate the conjugation of :code:`y` by the versor (k-reflection) :code:`x`,
    using the conjugation formula :math:`(-1)^{k \ell} x y x^{-1}`, where :math:`k` is the
    grade of :code:`x` and :math:`\ell` is the grade of the blade :code:`y`. (Eq 7.18 in [GA4CS]_)
    If :code:`y` is a multivector instead of a blade, the formula is applied to each pure
    grade component of :code:`y` separately to ensure a consistent result.
    **Important**: note that :code:`x` is assumed to be normalized such that :math:`x \widetilde{x} = 1`
    (i.e. :code:`x.normsq() == 1`). Moreover, grade preservation is enforced by the code.
    Expect unexpected results if this operator is used with non-versors.

    .. [GA4CS] Dorst, Lasenby, and Fontijne. Geometric Algebra for Computer Science. Morgan Kaufmann, 2007.

    :param x: The versor (k-reflection), i.e. a multivector satisfying :math:`x \widetilde{x} = 1`.
    :param y: The multivector to be conjugated.
    :return: tuple of keys in binary representation and a lambda function.
    :raises TypeError: If :code:`x` is not a versor (k-reflection) and thus neither even nor odd.
    """
    if len(set((g % 2 for g in x.grades))) != 1:
        raise TypeError("x must be a versor (k-reflection) and thus either even or odd.")
    xr = x.reverse()
    axar_scalar = (x * xr).grade(0)  # The scalar part of x * ~x, which is assumed to be 1.
    if max(x.grades) % 2 == 1:
        return sum(((x * (yg_involute := y.grade(g).involute()) * xr + yg_involute * (1 - axar_scalar)).grade(g) for g in y.grades), start=type(x)(x.algebra))
    return sum(((x * y.grade(g) * xr + y.grade(g) * (1 - axar_scalar)).grade(g) for g in y.grades), start=type(x)(x.algebra))


def codegen_cp(x, y):
    """
    Generate the commutator product of :code:`x` and :code:`y`: :code:`x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] - algebra.signs[ky, kx])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_acp(x, y):
    """
    Generate the anti-commutator product of :code:`x` and :code:`y`: :code:`x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] + algebra.signs[ky, kx])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_ip(x, y, diff_func=abs):
    """
    Generate the inner product of :code:`x` and :code:`y`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == diff_func(kx - ky)
    return codegen_product(x, y, filter_func=filter_func)


def codegen_lc(x, y):
    """
    Generate the left-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: -x)


def codegen_rc(x, y):
    """
    Generate the right-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: x)


def codegen_sp(x, y):
    """
    Generate the scalar product of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: 0)


def codegen_proj(x, y):
    fr"""
    Generate the projection of :code:`x` onto :code:`y`: :math:`(x \cdot y) \widetilde{y}`,
    where it is assumed that :code:`y` is a normalized versor (k-reflection) and hence :math:`y^{-1} = \widetilde{y}`.

    :param x: The multivector to be projected.
    :param y: The versor (k-reflection) onto which :code:`x` is projected.
    :return: tuple of keys in binary representation and a lambda function.
    :raises TypeError: If :code:`y` is not a versor (k-reflection).
    """
    if len(set((g % 2 for g in y.grades))) != 1:
        raise TypeError("y must be a versor (k-reflection) and thus either even or odd.")
    return (x | y) * y.reverse()


def codegen_op(x, y):
    """
    Generate the outer product of :code:`x` and :code:`y`: :code:`x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == kx + ky
    return codegen_product(x, y, filter_func=filter_func)


def codegen_rp(x, y):
    """
    Generate the regressive product of :code:`x` and :code:`y`:,
    :math:`x \\vee y`.

    :param x:
    :param y:
    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    key_pss = len(algebra) - 1
    keyout_func = lambda kx, ky: key_pss - (kx ^ ky)
    filter_func = lambda kx, ky, k_out: key_pss == kx + ky - k_out
    # Sign is composed of dualization of each blade, exterior product, and undual.
    sign_func = lambda pair: (
        algebra.signs[pair[0], key_pss - pair[0]] *
        algebra.signs[pair[1], key_pss - pair[1]] *
        algebra.signs[key_pss - pair[0], key_pss - pair[1]] *
        algebra.signs[key_pss - (pair[0] ^ pair[1]), pair[0] ^ pair[1]]
    )

    return codegen_product(
        x, y,
        filter_func=filter_func,
        keyout_func=keyout_func,
        sign_func=sign_func,
    )


Fraction = namedtuple('Fraction', ['numer', 'denom'])
Fraction.__doc__ = """
Tuple representing a fraction.
"""


def codegen_inv(y, symbolic=False):
    alg = y.algebra
    # If y * ~y is a scalar, use the simple blade inverse ~y / (y * ~y).
    # This matches GAmphetamine's check: if (gradeOf(a*~a) == 0) return gp(reverse(a), inv(sq))
    # and avoids producing unsimplified rational polynomials like (y * s) / s^2.
    yr = y.reverse()
    ynorm = y * yr
    if ynorm.grades == (0,):
        num = yr
        denom = ynorm
    elif alg.d < 6:
        num, denom = codegen_hitzer_inv(y, symbolic=True)
    else:
        num, denom = codegen_shirokov_inv(y, symbolic=True)

    if symbolic:
        return Fraction(num, denom)

    d = denom.e
    return num.map(lambda v: v / d)


def codegen_hitzer_inv(x, symbolic=False):
    """
    Generate code for the inverse of :code:`x` using the Hitzer inverse,
    which works up to 5D algebras.
    """
    alg = x.algebra
    d = alg.d
    if d == 0:
        num = alg.blades.e
    elif d == 1:
        num = x.involute()
    elif d == 2:
        num = x.conjugate()
    elif d == 3:
        xconj = x.conjugate()
        num = xconj * ~(x * xconj)
    elif d == 4:
        xconj = x.conjugate()
        x_xconj = x * xconj
        num = xconj * (x_xconj - 2 * x_xconj.grade(3, 4))
    elif d == 5:
        xconj = x.conjugate()
        x_xconj = x * xconj
        combo = xconj * ~x_xconj
        x_combo = x * combo
        num = combo * (x_combo - 2 * x_combo.grade(1, 4))
    else:
        raise NotImplementedError(f"Closed form inverses are not known in {d=} dimensions.")
    denom = x.sp(num)

    if symbolic:
        return Fraction(num, denom)
    denom = denom.e
    return num.map(lambda v: v / denom)


def codegen_shirokov_inv(x, symbolic=False):
    """
    Generate code for the inverse of :code:`x` using the Shirokov inverse,
    which is works in any algebra, but it can be expensive to compute.
    """
    alg = x.algebra
    n = 2 ** ((alg.d + 1) // 2)
    supply = power_supply(x, tuple(range(1, n + 1)))  # Generate powers of x efficiently.
    powers = []
    cs = []
    xs = []
    for i in range(1, n + 1):
        powers.append(next(supply))
        xi = powers[i - 1]
        for j in range(i - 1):
            power_idx = i - j - 2
            xi_diff = powers[power_idx] * cs[j]
            xi = xi - xi_diff
        if xi.grades == (0,):
            break
        xs.append(xi)
        cs.append(s if (s := xi.e) == 0 else n * s / i)

    if i == 1:
        adj = alg.blades.e
    else:
        adj = xs[-1] - cs[-1]

    if symbolic:
        return Fraction(adj, xi)
    xi = xi.e
    return adj.map(lambda v: v / xi)


def codegen_div(x, y):
    """
    Generate code for :math:`x y^{-1}`.
    """
    num, denom = codegen_inv(y, symbolic=True)
    if not denom:
        raise ZeroDivisionError
    d = denom.e
    return (x * num).map(lambda v: v / d)


def codegen_normsq(x):
    return x * ~x


def codegen_outerexp(x, asterms=False):
    alg = x.algebra
    if len(x.grades) != 1:
        warnings.warn('Outer exponential might not converge for mixed-grade multivectors.', RuntimeWarning)
    k = alg.d

    Ws = [alg.scalar(e=1), x]
    j = 2
    while j <= k:
        Wj = Ws[-1] ^ x
        # Dividing like this avoids floating point numbers, which is excellent.
        Wj._values = tuple(v / j for v in Wj._values)
        if Wj:
            Ws.append(Wj)
            j += 1
        else:
            break

    if asterms:
        return Ws
    return reduce(operator.add, Ws)

def codegen_outersin(x):
    odd_Ws = codegen_outerexp(x, asterms=True)[1::2]
    outersin = reduce(operator.add, odd_Ws)
    return outersin


def codegen_outercos(x):
    even_Ws = codegen_outerexp(x, asterms=True)[0::2]
    outercos = reduce(operator.add, even_Ws)
    return outercos


def codegen_outertan(x):
    Ws = codegen_outerexp(x, asterms=True)
    even_Ws, odd_Ws = Ws[0::2], Ws[1::2]
    outercos = reduce(operator.add, even_Ws)
    outersin = reduce(operator.add, odd_Ws)
    outertan = outersin / outercos
    return outertan


def codegen_add(x, y):
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] + v
        else:
            vals[k] = v
    return {k: v for k, v in vals.items() if v != 0}


def codegen_sub(x, y):
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] - v
        else:
            vals[k] = -v
    return {k: v for k, v in vals.items() if v != 0}

def codegen_neg(x):
    return {k: -v for k, v in x.items()}


def codegen_involutions(x, invert_grades=(2, 3)):
    """
    Codegen for the involutions of Clifford algebras:
    reverse, grade involute, and Clifford involution.

    :param invert_grades: The grades that flip sign under this involution mod 4, e.g. (2, 3) for reversion.
    """
    return {k: -v if bin(k).count('1') % 4 in invert_grades else v
            for k, v in x.items()}


def codegen_reverse(x):
    return codegen_involutions(x, invert_grades=(2, 3))


def codegen_involute(x):
    return codegen_involutions(x, invert_grades=(1, 3))


def codegen_conjugate(x):
    return codegen_involutions(x, invert_grades=(1, 2))


def codegen_sqrt(x):
    """
    Take the square root using the study number approach as described in
    https://doi.org/10.1002/mma.8639
    """
    alg = x.algebra
    if x.grades == (0,):
        return x.map(lambda v: v**0.5)
    a, bI = x.grade(0), x - x.grade(0)
    has_solution = len(x.grades) <= 2 and 0 in x.grades
    if not has_solution:
        warnings.warn("Cannot verify that we really are taking the sqrt of a Study number.", RuntimeWarning)

    bI_sq = bI * bI
    if not bI_sq:
        cp = a.e**0.5
    else:
        normS = (a * a - bI_sq).e
        cp = (0.5 * (a.e + normS**0.5))**0.5
    return (0.5 * bI / cp) + cp


def codegen_polarity(x, undual=False):
    if undual:
        return x * x.algebra.pss
    key_pss = len(x.algebra) - 1
    sign = x.algebra.signs[key_pss, key_pss]
    if sign == -1:
        return - x * x.algebra.pss
    return codegen_gp(x, x.algebra.pss)


def codegen_unpolarity(x):
    return codegen_polarity(x, undual=True)


def codegen_hodge(x, undual=False):
    if undual:
        return {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[key_dual, eI] < 0 else v
                for eI, v in x.items()}
    return {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[eI, key_dual] < 0 else v
            for eI, v in x.items()}


def codegen_unhodge(x):
    return codegen_hodge(x, undual=True)


def _lambdify_mv(mv):
    func = lambdify(
        args={'x': sorted(mv.free_symbols, key=lambda x: x.name)},
        exprs=list(mv.values()),
        funcname=f'custom_{mv.type_number}',
        cse=mv.algebra.cse
    )
    return CodegenOutput(tuple(mv.keys()), func)


def resolve_layout(layouts: dict, res_layout: dict, MVType: type = None):
    """
    Look up the best-matching MVType for a given result layout from a set of registered types.

    :param layouts: mapping from MVType (class) to a layout dict. A layout is a
        dict from blade key (integer) to either ``...`` for a free component, or
        a number for a fixed constant (e.g. the homogeneous coordinate ``1.0``
        of a point).
    :param res_layout: the layout dict of the result whose type we are trying to
        identify, in the same ``{key: ... | number}`` form.
    :param MVType: optional class used to restrict the search to that type and
        its subclasses (e.g. to prefer a more specific ``NormalizedPoint`` over
        a generic ``Point`` when the type of the result is already partially
        known). Requires the keys of ``layouts`` to be classes.
    :return: ``(cls, layout)`` for the best match, or ``(None, None)`` if no
        registered type matches.

    A registered type is considered a *match* for the result if:

    - all fixed constants in the type's layout agree with the result
      (no conflicting fixed values, no fixed blades absent from the result);
    - all free components in the result are also free in the type's layout
      (the type doesn't fix something the result leaves open);
    - all structural constants in the result are covered by the type's layout
      (the type must know about every fixed blade the result carries).

    When multiple types match, the most specific one wins: first minimising the
    number of free slots in the registered layout that coincide with fixed values
    in the result (tighter structural match), then minimising free slots that fall
    outside the result entirely (smaller footprint). Ties are broken by
    registration order in ``layouts``.
    """
    res_free = {k for k, v in res_layout.items() if v is Ellipsis}
    res_fixed_keys = {k for k, v in res_layout.items() if v is not Ellipsis}
    res_fixed_items = {(k, v) for k, v in res_layout.items() if v is not Ellipsis}
    res_keys = res_free | res_fixed_keys

    best, best_cost = (None, None), None
    for cls, L in layouts.items():
        if MVType is not None and not issubclass(cls, MVType):
            continue
        free = {k for k, v in L.items() if v is Ellipsis}
        fixed_items = {(k, v) for k, v in L.items() if v is not Ellipsis}
        all_keys = L.keys()
        if not res_free.issubset(free):
            continue
        if not fixed_items.issubset(res_fixed_items):
            continue
        if not res_fixed_keys.issubset(all_keys):
            continue
        cost = (len(free & res_fixed_keys), len(free - res_keys))
        if best_cost is None or cost < best_cost:
            best, best_cost = (cls, L), cost
            if cost == (0, 0):
                break  # perfect fit; layouts are iterated in insertion order so this is optimal

    return best


def do_codegen(codegen, *mvs, printer=None, func_printer=None) -> CodegenOutput:
    """
    :param codegen: callable that performs codegen for the given :code:`mvs`. This can be any callable
        that returns either a :class:`~kingdon.multivector.MultiVector`, a dictionary, or an instance of :class:`CodegenOutput`.
    :param mvs: Any remaining positional arguments are taken to be symbolic :class:`~kingdon.multivector.MultiVector`'s.
    :param printer: The sympy style printer used to generate the code with sympy-style printing.
    :param func_printer: The sympy style evaluator printer used to generate the code with sympy-style printing.
    :return: Instance of :class:`CodegenOutput`.
    """
    algebra = mvs[0].algebra
    mvs_orig = [copy.deepcopy(mv) for mv in mvs]

    res = codegen(*(mv.asmvtype() for mv in mvs))

    output_mv_idx = None  # If codegen modified one of the mvs using set, this will be the index of the modified mv.
    if res is None:
        output_mv_idx = next(i for i, mv in enumerate(mvs) if mv != mvs_orig[i])
        res = mvs[output_mv_idx]
        mvs = mvs_orig
    else:
        def is_number(x):
            try: float(x); return True
            except (ValueError, TypeError): return False
        res_layout = {k: v if is_number(str(v)) else ... for k, v in res.items()}
        MVType, layout = resolve_layout(algebra._type_layouts, res_layout)
        
        if layout is not None:
            res = {k: v for k, v in res.items() if layout[k] == ...}
            
    funcname = f'{codegen.__name__}_' + '_x_'.join(f"{format(mv[0].type_number if isinstance(mv, list) else mv.type_number, 'X')}" for mv in mvs)
    args = {arg_name: [tuple(chain(*(x.values() for x in arg)))] if isinstance(arg, list) else arg.values()
            for arg_name, arg in zip(string.ascii_uppercase, mvs)}

    # Sort the keys in canonical order
    res = {bin: res[bin] if isinstance(res, dict) else getattr(res, canon)
           for canon, bin in algebra.canon2bin.items() if bin in res.keys()}

    if all(isinstance(v, str) for v in res.values()):
        return func_builder(res, *mvs, args=args, funcname=funcname, MVType=MVType)  # TODO: add output_mv_idx support

    keys, exprs = tuple(res.keys()), list(res.values())
    if output_mv_idx is not None:
        keys = ()
    func = lambdify(args, exprs, funcname=funcname,
                    cse=algebra.cse, printer=printer, func_printer=func_printer,
                    output_mv_idx=output_mv_idx
                    )
    func.output_mv_idx = output_mv_idx
    return CodegenOutput(
        keys, func, MVType
    )

def do_compile(codegen, *tapes):
    algebra = tapes[0].algebra
    namespace = algebra.numspace

    res = codegen(*tapes)
    funcname = f'{codegen.__name__}_' + '_x_'.join(f"{tape.type_number}" for tape in tapes)
    funcstr = f"def {funcname}({', '.join(t.expr for t in tapes)}):"
    if not isinstance(res, str):
        funcstr += f"    return {res.expr}"
    else:
        funcstr += f"    return ({res},)"

    funclocals = {}
    filename = f'<{funcname}>'
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    func = funclocals[funcname]
    return CodegenOutput(
        res.keys() if not isinstance(res, str) else (0,), func
    )


def _count_muls_adds(funcstr: str) -> tuple:
    """Count multiplication and addition/subtraction operations in a generated function string.

    :return: Tuple of (muls, adds).
    """
    muls = funcstr.count('*')
    adds = funcstr.count('+') + funcstr.count('-')
    return muls, adds


def _build_and_cache_func(header, body_lines, funcname, namespace=None):
    """Build a function from header + body lines, insert op-count docstring, compile, exec, cache.

    :param header: The `def funcname(...):` line.
    :param body_lines: List of indented body lines (without the docstring).
    :param funcname: Name used as the linecache key.
    :param namespace: Execution namespace dict. Defaults to {'builtins': builtins, 'range': range}.
    :return: The compiled function object.
    """
    if namespace is None:
        namespace = {'builtins': builtins, 'range': range}
    func_source_no_doc = header + '\n' + '\n'.join(body_lines)
    muls, adds = _count_muls_adds(func_source_no_doc)
    all_lines = [header, f'    """{muls} muls / {adds} adds"""'] + body_lines
    func_source = '\n'.join(all_lines)
    func_locals = {}
    exec(compile(func_source, funcname, 'exec'), namespace, func_locals)
    linecache.cache[funcname] = (len(func_source), None, func_source.splitlines(True), funcname)
    return func_locals[funcname]


def func_builder(res_vals: defaultdict, *mvs, args: dict, funcname: str, MVType: Type = None) -> CodegenOutput:
    """
    Build a Python function for the product between given multivectors.

    :param res_vals: Dict to be converted into a function. The keys correspond to the basis blades in binary,
        while the values are strings to be converted into source code.
    :param mvs: all the multivectors that the resulting function is a product of.
    :param funcname: Name of the function. Be aware: if a function by that name already existed, it will be overwritten.
    :return: tuple of output keys of the callable, and the callable.
    """
    header = f'def {funcname}({", ".join(args.keys())}):'
    body_lines = []
    if res_vals:
        for name, arg in args.items():
            body_lines.append(f'    [{", ".join(str(v) for v in arg)}] = {name}')
        body_lines.append(f'    return [{", ".join(res_vals.values())},]')
    else:
        body_lines.append(f'    return list()')
    func = _build_and_cache_func(header, body_lines, funcname, namespace={})
    return CodegenOutput(tuple(res_vals.keys()), func, MVType)


def _poly_cse_compute(exprs: List[RationalPolynomial], common_denom: Optional[Polynomial] = None):
    """
    Run CSE on a list of :class:`~kingdon.polynomial.RationalPolynomial` expressions.

    :param exprs: list of :class:`~kingdon.polynomial.RationalPolynomial` expressions.
    :param common_denom: optional :class:`~kingdon.polynomial.Polynomial` common denominator.
    :return: (cse_pairs, numer_simplified, denom_simplified) where:
        - cse_pairs: list of (name, poly_args) tuples for each extracted subexpression.
        - numer_simplified: list of poly_args lists for simplified numerators.
        - denom_simplified: poly_args list for the simplified denominator, or None.
    """
    # Build CSE input: numerators of all exprs, plus the common denominator as last entry.
    poly_args_list = [e.numer.args for e in exprs]
    if common_denom is not None:
        poly_args_list.append(common_denom.args)

    all_vars = {f for pl in poly_args_list for m in pl for f in m[1:] if isinstance(f, str)}
    cse_pairs, simplified = poly_cse(poly_args_list, prot=None, iso=[2] + sorted(all_vars))

    numer_simplified = simplified[:-1] if common_denom is not None else simplified
    denom_simplified = simplified[-1] if common_denom is not None else None

    return cse_pairs, numer_simplified, denom_simplified


def _rp_var_name(v):
    """Return the variable name string for a simple :class:`~kingdon.polynomial.RationalPolynomial` symbol, or ``'_'``."""
    numer_args = getattr(getattr(v, 'numer', None), 'args', None)
    if (numer_args and len(numer_args) == 1
            and len(numer_args[0]) == 2
            and numer_args[0][0] == 1):
        return str(numer_args[0][1])
    return '_'


def unflatten(template, flat):
    it = iter(flat)
    def walk(t):
        return type(t)(walk(x) for x in t) if isinstance(t, (list, tuple)) else next(it)
    return walk(template)


def _lambdify_poly_cse(args_dict, exprs, funcname, cse_pairs, numer_simplified, denom_simplified, output_mv_idx=None):
    """
    Build a Python function from pre-computed polynomial CSE results.

    :param args_dict: dict mapping arg name (str) to list of :class:`~kingdon.polynomial.RationalPolynomial` values.
    :param exprs: list of :class:`~kingdon.polynomial.RationalPolynomial` expressions (for denom checks).
    :param funcname: name for the generated function.
    :param cse_pairs: list of (name, poly_args) from :func:`_poly_cse_compute`.
    :param numer_simplified: simplified numerator poly_args per expression.
    :param denom_simplified: simplified denominator poly_args, or None.
    :param output_mv_idx: index into the argument list of the MV to write the result into (for set-style codegen).
    :return: compiled function with docstring containing op counts.
    """
    names = list(args_dict)
    body_lines = []
    for name, values in args_dict.items():
        has_nested = any(isinstance(v, (list, tuple)) for v in values)
        if has_nested:
            temp_names = [f'_{name}_{i}' for i in range(len(values))]
            body_lines.append(f'    [{", ".join(temp_names)}] = {name}')
            for temp_name, v in zip(temp_names, values):
                if isinstance(v, (list, tuple)):
                    body_lines.append(f'    [{", ".join(_rp_var_name(sv) for sv in v)}] = {temp_name}')
                else:
                    body_lines.append(f'    {_rp_var_name(v)} = {temp_name}')
        else:
            body_lines.append(f'    [{", ".join(_rp_var_name(v) for v in values)}] = {name}')

    for cse_name, poly_args in cse_pairs:
        body_lines.append(f'    {cse_name}={poly_format(poly_args)}')

    # Emit denominator local variable if needed (avoids recomputing it per return component)
    if denom_simplified is not None and sum(1 for e in exprs if e.denom != 1) > 1:
        cse_names = {cse_name for cse_name, _ in cse_pairs}
        denom_var = '_d'
        while denom_var in cse_names:
            denom_var += '_'
        body_lines.append(f'    {denom_var}={poly_format(denom_simplified)}')
        denom_ref = denom_var
    else:
        denom_ref = poly_format(denom_simplified) if denom_simplified is not None else None

    ret_parts = [
        poly_format(simp) if (denom_ref is None or e.denom == 1)
        else f'({poly_format(simp)})/({denom_ref})'
        for e, simp in zip(flatten(exprs), numer_simplified)
    ]
    ret_parts = unflatten(exprs, ret_parts)
    if output_mv_idx is not None:
        output_name = names[output_mv_idx]
        for i, part in enumerate(ret_parts):
            body_lines.append(f'    {output_name}[{i}] = {str(part).replace("'", "")}')
        body_lines.append('    return ()')
    else:
        body_lines.append(f'    return {str(ret_parts).replace("'", "")}')

    header = f'def {funcname}({", ".join(names)}):'
    return _build_and_cache_func(header, body_lines, funcname)

def lambdify(
        args: dict,
        exprs: list,
        funcname: str,
        printer=None,
        func_printer=None,
        cse=False,
        output_mv_idx: int = None,
    ):
    """
    Function that turns symbolic expressions into Python functions. Heavily inspired by
    :mod:`sympy`'s function by the same name, but adapted for the needs of :code:`kingdon`.

    Particularly, this version gives us more control over the names of the function and its
    arguments, and is more performant, particularly when the given expressions are strings.

    Example usage:

    .. code-block ::

        alg = Algebra(2)
        a = alg.multivector(name='a')
        b = alg.multivector(name='b')
        args = {'A': a.values(), 'B': b.values()}
        exprs = tuple(codegen_cp(a, b).values())
        func = lambdify(args, exprs, funcname='cp', cse=False)

    This will produce the following code:

    .. code-block ::

        def cp(A, B):
            [a, a1, a2, a12] = A
            [b, b1, b2, b12] = B
            return (+a1*b2-a2*b1,)

    .. note::
        As a `kingdon` end user, you should probably not need to call this functon directly,
        be sure to check out :meth:`~kingdon.algebra.Algebra.register` first.
        And even for experienced users or `kingdon` developers it is recommended
        to use :func:`do_codegen` which provides a clean API around this function.

    :param args: dictionary of type dict[str | Symbol, tuple[Symbol]].
    :param exprs: tuple[Expr]
    :param funcname: string to be used as the bases for the name of the function.
    :param printer: Instance of the sympy style printer used to print individual sympy expressions.
    :param func_printer: Instance of the sympy style printer used to generate functions using the `printer`.
    :param cse: If :code:`True` (default), CSE is applied to the expressions.
        This typically greatly improves performance and reduces numba's initialization time.
    :param output_mv_idx: Index of the multivector that stores the result returned by the codegen function.
        If :code:`None`, the generated function will return the values of the multivector.
    :return: Function that represents that can be used to calculate the values of exprs.
    """
    cses, _exprs = [], exprs
    cse_pairs, numer_simplified, denom_simplified = None, None, None

    flattened_exprs = flatten(exprs)
    if exprs and all(isinstance(e, RationalPolynomial) for e in flattened_exprs):
        if cse:
            non_unit = [e for e in flattened_exprs if e.denom != 1]
            if not non_unit or all(e.denom == non_unit[0].denom for e in non_unit):
                common_denom = non_unit[0].denom if non_unit else None
                cse_pairs, numer_simplified, denom_simplified = _poly_cse_compute(flattened_exprs, common_denom)

                if printer is None and func_printer is None:
                    return _lambdify_poly_cse(args, exprs, funcname, cse_pairs, numer_simplified, denom_simplified,
                                              output_mv_idx=output_mv_idx)

    tosympy = lambda x: x.tosympy() if hasattr(x, 'tosympy') else x
    if cse_pairs is not None:
        args = {name: [tosympy(v) for v in values] for name, values in args.items()}
        cses = [(name, tosympy(Polynomial(poly_args))) for name, poly_args in cse_pairs]
        numer_syms = [tosympy(Polynomial(expr)) for expr in numer_simplified]
        denom_sym = tosympy(Polynomial(denom_simplified)) if denom_simplified is not None else None
        _exprs = [
            numer if (denom_sym is None or e.denom == 1) else numer / denom_sym
            for e, numer in zip(exprs, numer_syms)
        ]
    else:
        args = {name: [tosympy(v) for v in values] for name, values in args.items()}
        _exprs = [tosympy(expr) for expr in exprs]

    if cse and not cses:
        if not callable(cse):
            from sympy.simplify.cse_main import cse
        cses, _exprs = cse(_exprs, list=False)

    if not any(_exprs):
        _exprs = list('0' for expr in _exprs)

    if printer is None:
        printer = LambdaPrinter(
            {'fully_qualified_modules': False, 'inline': True,
             'allow_unknown_functions': True,
             'user_functions': {}}
        )
    if func_printer is None:
        func_printer = KingdonPrinter(printer)

    names = tuple(arg if isinstance(arg, str) else arg.name for arg in args.keys())
    iterable_args = tuple(args.values())
    funcstr = func_printer.doprint(funcname, iterable_args, names, _exprs, cses=cses, output_mv_idx=output_mv_idx)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace = {'builtins': builtins, 'range': range, **(printer.namespace if hasattr(printer, 'namespace') else {})}

    funclocals = {}
    filename = f'<{funcname}>'
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    func = funclocals[funcname]
    func.__module__ = __name__
    return func


class KingdonPrinter:
    def __init__(self, printer=None, dummify=False):
        self._dummify = dummify

        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import LambdaPrinter

        if printer is None:
            printer = LambdaPrinter()

        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            if inspect.isclass(printer):
                printer = printer()

            self._exprrepr = printer.doprint

        # Used to print the generated function arguments in a standard way
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, names, expr, *, cses=(), output_mv_idx=None):
        """
        Returns the function definition code as a string.
        """
        funcbody = []

        if not iterable(args):
            args = [args]

        if cses:
            subvars, subexprs = zip(*cses)
            exprs = [expr] + list(subexprs)
            argstrs, exprs = self._preprocess(args, exprs)
            expr, subexprs = exprs[0], exprs[1:]
            cses = zip(subvars, subexprs)
        else:
            argstrs, expr = self._preprocess(args, expr)

        # Generate argument unpacking and final argument list
        funcargs = []
        unpackings = []

        for i, (name, argstr, arg) in enumerate(zip(names, argstrs, args)):
            if not arg:
                funcargs.append(name)
            elif iterable(argstr):
                funcargs.append(name)
                if i == output_mv_idx: continue
                if iterable(argstr[0]):
                    unpackings.extend(self._print_unpacking([f'{name}_{i}' for i in range(len(argstr))], name))
                    for i, subargstr in enumerate(argstr):
                        unpackings.extend(self._print_unpacking(subargstr, f'{name}_{i}'))
                else:
                    unpackings.extend(self._print_unpacking(argstr, name))
            else:
                funcargs.append(argstr)

        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # Wrap input arguments before unpacking
        funcbody.extend(self._print_funcargwrapping(funcargs))

        funcbody.extend(unpackings)

        for s, e in cses:
            if e is None:
                funcbody.append('del {}'.format(s))
            else:
                funcbody.append('{} = {}'.format(s, self._exprrepr(e)))

        if output_mv_idx is not None:
            for i, e in enumerate(expr):
                e_str = _recursive_to_string(self._exprrepr, e)
                funcbody.append(f'{names[output_mv_idx]}[{i}] = ({e_str})' if '\n' in e_str else f'{names[output_mv_idx]}[{i}] = {e_str}')
            funcbody.append('return ()')
        else:
            str_expr = _recursive_to_string(self._exprrepr, expr)
            if '\n' in str_expr:
                str_expr = '({})'.format(str_expr)
            funcbody.append('return {}'.format(str_expr))

        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])
        funcstr = '\n'.join(funclines) + '\n'
        muls, adds = _count_muls_adds(funcstr)
        funclines.insert(1, f'    """{muls} muls / {adds} adds"""')

        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        return isinstance(ident, str) and ident.isidentifier() \
                and not keyword.iskeyword(ident)

    def _preprocess(self, args, expr):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        argstrs = [None]*len(args)
        for i, arg in enumerate(args):
            if iterable(arg):
                s, expr = self._preprocess(arg, expr)
            elif hasattr(arg, 'free_symbols') and not arg.free_symbols:
                # sympy constant (no free symbols): use _ as placeholder in unpacking
                s = '_'
            elif hasattr(arg, 'name'):
                s = arg.name
            elif hasattr(arg, 'is_symbol') and arg.is_symbol:
                s = self._argrepr(arg)
            else:
                s = str(arg)
            argstrs[i] = s
        return argstrs, expr

    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
        return []

    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
        def unpack_lhs(lvalues):
            return '({},)'.format(', '.join(
                unpack_lhs(val) if iterable(val) else val for val in lvalues))

        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]

def _recursive_to_string(doprint, arg):
    if isinstance(arg, str):
        return arg
    elif not arg:
        return str(arg)  # Empty list or tuple
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ",)"
        else:
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        return ''.join((left, ', '.join(_recursive_to_string(doprint, e) for e in arg), right))
    else:
        return doprint(arg)
