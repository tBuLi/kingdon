"""
Built-in geometric operators, such as the geometric product, wedge product, sandwich, inverse, etc.

These functions are assumed to operate on multivectors, and return a multivector.
"""
from itertools import product
from collections import namedtuple
import warnings
import operator
from typing import Callable
from functools import reduce, wraps

from kingdon.multivector import MultiVector
from kingdon.powers import power_supply


def dict_to_multivector(res: dict, algebra) -> MultiVector:
    res = {k: v for k, v in res.items() if v}
    keys, values = zip(*res.items()) if res else ((), [])
    return MultiVector.fromkeysvalues(algebra, keys, values)


def codegen_product(
    x: MultiVector,
    y: MultiVector,
    filter_func=None,
    sign_func=None,
    keyout_func=operator.xor
) -> MultiVector:
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

    res = {k: 0 for k in x.algebra.canon2bin.values()}
    for (kx, vx), (ky, vy) in product(x.items(), y.items()):
        if (sign := sign_func((kx, ky))):
            key_out = keyout_func(kx, ky)
            if filter_func and not filter_func(kx, ky, key_out): continue
            termstr = vx * vy if sign > 0 else (- vx * vy)
            res[key_out] += termstr
    return dict_to_multivector(res, x.algebra)


def codegen_gp(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    return codegen_product(x, y)


def codegen_sw(x: MultiVector, y: MultiVector) -> MultiVector:
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
    condition = 1 - (x * xr).grade(0)  # The scalar part of x * ~x is assumed to be 1.
    if max(x.grades) % 2 == 1:
        return sum(((x * (yg_involute := y.grade(g).involute()) * xr + yg_involute * condition).grade(g) for g in y.grades), start=type(x)(x.algebra))
    return sum(((x * y.grade(g) * xr + y.grade(g) * condition).grade(g) for g in y.grades), start=type(x)(x.algebra))


def codegen_cp(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the commutator product of :code:`x` and :code:`y`: :code:`x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] - algebra.signs[ky, kx])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_acp(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the anti-commutator product of :code:`x` and :code:`y`: :code:`x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] + algebra.signs[ky, kx])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_ip(x: MultiVector, y: MultiVector, diff_func: Callable=abs) -> MultiVector:
    """
    Generate the inner product of :code:`x` and :code:`y`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == diff_func(kx - ky)
    return codegen_product(x, y, filter_func=filter_func)


def codegen_lc(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the left-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: -x)


def codegen_rc(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the right-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: x)


def codegen_sp(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the scalar product of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return codegen_ip(x, y, diff_func=lambda x: 0)


def codegen_proj(x: MultiVector, y: MultiVector) -> MultiVector:
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
    return ((x | y) * y.reverse() + (x * (1 - y.normsq().grade(0)))).grade(x.grades)


def codegen_op(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate the outer product of :code:`x` and :code:`y`: :code:`x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == kx + ky
    return codegen_product(x, y, filter_func=filter_func)


def codegen_rp(x: MultiVector, y: MultiVector) -> MultiVector:
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


def codegen_inv(y: MultiVector, symbolic: bool = False) -> MultiVector:
    alg = y.algebra
    # If y * ~y is a scalar, use the simple blade inverse ~y / (y * ~y).
    # This matches GAmphetamine's check: if (gradeOf(a*~a) == 0) return gp(reverse(a), inv(sq))
    # and avoids producing unsimplified rational polynomials like (y * s) / s^2.
    yr = y.reverse()
    ynorm = y.normsq()
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


def codegen_hitzer_inv(x: MultiVector, symbolic: bool = False) -> MultiVector:
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


def codegen_shirokov_inv(x: MultiVector, symbolic: bool = False) -> MultiVector:
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


def codegen_div(x: MultiVector, y: MultiVector) -> MultiVector:
    """
    Generate code for :math:`x y^{-1}`.
    """
    num, denom = codegen_inv(y, symbolic=True)
    if not denom:
        raise ZeroDivisionError
    d = denom.e
    return (x * num).map(lambda v: v / d)


def codegen_normsq(x: MultiVector) -> MultiVector:
    return x * ~x


def codegen_outerexp(x: MultiVector, asterms: bool = False) -> MultiVector:
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


def codegen_outersin(x: MultiVector) -> MultiVector:
    odd_Ws = codegen_outerexp(x, asterms=True)[1::2]
    outersin = reduce(operator.add, odd_Ws)
    return outersin


def codegen_outercos(x: MultiVector) -> MultiVector:
    even_Ws = codegen_outerexp(x, asterms=True)[0::2]
    outercos = reduce(operator.add, even_Ws)
    return outercos


def codegen_outertan(x: MultiVector) -> MultiVector:
    Ws = codegen_outerexp(x, asterms=True)
    even_Ws, odd_Ws = Ws[0::2], Ws[1::2]
    outercos = reduce(operator.add, even_Ws)
    outersin = reduce(operator.add, odd_Ws)
    outertan = outersin / outercos
    return outertan


def codegen_add(x: MultiVector, y: MultiVector) -> MultiVector:
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] + v
        else:
            vals[k] = v
    return dict_to_multivector(vals, x.algebra)


def codegen_sub(x: MultiVector, y: MultiVector) -> MultiVector:
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] - v
        else:
            vals[k] = -v
    return dict_to_multivector(vals, x.algebra)


def codegen_neg(x: MultiVector) -> MultiVector:
    return dict_to_multivector({k: -v for k, v in x.items()}, x.algebra)


def codegen_involutions(x: MultiVector, invert_grades: tuple[int, int] = (2, 3)) -> MultiVector:
    """
    Codegen for the involutions of Clifford algebras:
    reverse, grade involute, and Clifford involution.

    :param invert_grades: The grades that flip sign under this involution mod 4, e.g. (2, 3) for reversion.
    """
    res = {k: -v if bin(k).count('1') % 4 in invert_grades else v
           for k, v in x.items()}
    return dict_to_multivector(res, x.algebra)


def codegen_reverse(x: MultiVector) -> MultiVector:
    return codegen_involutions(x, invert_grades=(2, 3))


def codegen_involute(x: MultiVector) -> MultiVector:
    return codegen_involutions(x, invert_grades=(1, 3))


def codegen_conjugate(x: MultiVector) -> MultiVector:
    return codegen_involutions(x, invert_grades=(1, 2))


def codegen_sqrt(x: MultiVector) -> MultiVector:
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


def codegen_polarity(x: MultiVector, undual: bool = False) -> MultiVector:
    if undual:
        return x * x.algebra.pss
    key_pss = len(x.algebra) - 1
    sign = x.algebra.signs[key_pss, key_pss]
    if sign == -1:
        return - x * x.algebra.pss
    return codegen_gp(x, x.algebra.pss)


def codegen_unpolarity(x: MultiVector) -> MultiVector:
    return codegen_polarity(x, undual=True)


def codegen_hodge(x: MultiVector, undual: bool = False) -> MultiVector:
    if undual:
        res = {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[key_dual, eI] < 0 else v
               for eI, v in x.items()}
    else:
        res = {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[eI, key_dual] < 0 else v
               for eI, v in x.items()}
    keys, values = zip(*res.items())
    return MultiVector.fromkeysvalues(x.algebra, keys, values)


def codegen_unhodge(x: MultiVector) -> MultiVector:
    return codegen_hodge(x, undual=True)
