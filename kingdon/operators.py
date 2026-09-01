"""
Built-in geometric operators, such as the geometric product, wedge product, sandwich, inverse, etc.

These functions are assumed to operate on multivectors, and return a multivector.
"""
import itertools
from collections import namedtuple
import warnings
import operator
from collections.abc import Callable
from functools import reduce, wraps
from fractions import Fraction as PyFraction

from kingdon.powers import power_supply


def dict_to_multivector(res: dict, algebra) -> "MultiVector":
    from kingdon.multivector import MultiVector  # TODO: Could perhaps be avoided by passing as items?
    # Drop zeros and put the remaining keys back in canon2bin order.
    nonzero = {k: v for k, v in res.items() if v}
    items = [(k, nonzero[k]) for k in algebra.canon2bin.values() if k in nonzero]
    keys, values = zip(*items) if items else ((), [])
    return MultiVector.fromkeysvalues(algebra, keys, list(values), raw=True)


def product(
    x: "MultiVector",
    y: "MultiVector",
    filter_func=None,
    sign_func=None,
    keyout_func=operator.xor
) -> "MultiVector":
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
    for (kx, vx), (ky, vy) in itertools.product(x.items(), y.items()):
        if (sign := sign_func((kx, ky))):
            key_out = keyout_func(kx, ky)
            if filter_func and not filter_func(kx, ky, key_out): continue
            termstr = vx * vy if sign > 0 else (- vx * vy)
            res[key_out] += termstr
    return dict_to_multivector(res, x.algebra)


def gp(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    return product(x, y)


def sw(x: "MultiVector", y: "MultiVector") -> "MultiVector":
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
    xr = reverse(x)
    condition = sub(xr.algebra.multivector(e=1), grade(gp(x, xr), 0))  # The scalar part of x * ~x is assumed to be 1.
    empty_mv = type(x)(x.algebra)
    if max(x.grades) % 2 == 1:
        return sum((grade((add(gp(x, gp((yg_involute := involute(grade(y, g))), xr)), gp(yg_involute, condition))), g) for g in y.grades), start=empty_mv)
    return sum((grade((add(gp(x, gp(grade(y, g), xr)), gp(grade(y, g), condition))), g) for g in y.grades), start=empty_mv)


def cp(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the commutator product of :code:`x` and :code:`y`: :code:`x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] - algebra.signs[ky, kx])
    return product(x, y, filter_func=filter_func)


def acp(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the anti-commutator product of :code:`x` and :code:`y`: :code:`x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda kx, ky, k_out: (algebra.signs[kx, ky] + algebra.signs[ky, kx])
    return product(x, y, filter_func=filter_func)


def ip(x: "MultiVector", y: "MultiVector", diff_func: Callable=abs) -> "MultiVector":
    """
    Generate the inner product of :code:`x` and :code:`y`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == diff_func(kx - ky)
    return product(x, y, filter_func=filter_func)


def lc(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the left-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return ip(x, y, diff_func=lambda x: -x)


def rc(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the right-contraction of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return ip(x, y, diff_func=lambda x: x)


def sp(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the scalar product of :code:`x` and :code:`y`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return ip(x, y, diff_func=lambda x: 0)


def proj(x: "MultiVector", y: "MultiVector") -> "MultiVector":
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
    condition = sub(y.algebra.multivector(e=1), grade(normsq(y), 0))  # The scalar part of x * ~x is assumed to be 1.
    return grade(add(gp(ip(x, y), reverse(y)), gp(x, condition)), x.grades)


def op(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate the outer product of :code:`x` and :code:`y`: :code:`x.op(y) = x ^ y`.

    :x: "MultiVector"
    :y: "MultiVector"
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    filter_func = lambda kx, ky, k_out: k_out == kx + ky
    return product(x, y, filter_func=filter_func)


def rp(x: "MultiVector", y: "MultiVector") -> "MultiVector":
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

    return product(
        x, y,
        filter_func=filter_func,
        keyout_func=keyout_func,
        sign_func=sign_func,
    )

def grade(x: "MultiVector", *grades) -> "MultiVector":
    """ Select grade g part of x. """
    if len(grades) == 1 and isinstance(grades[0], tuple):
        grades = grades[0]
    res = {k: v for k, v in x.items() if k.bit_count() in grades}
    return dict_to_multivector(res, x.algebra)


Fraction = namedtuple('Fraction', ['numer', 'denom'])
Fraction.__doc__ = """
Tuple representing a fraction.
"""


def inv(y: "MultiVector", symbolic: bool = False) -> "MultiVector":
    alg = y.algebra
    # If y * ~y is a scalar, use the simple blade inverse ~y / (y * ~y).
    # This matches GAmphetamine's check: if (gradeOf(a*~a) == 0) return gp(reverse(a), inv(sq))
    # and avoids producing unsimplified rational polynomials like (y * s) / s^2.
    yr = reverse(y)
    ynorm = normsq(y)
    if ynorm.grades == (0,):
        num = yr
        denom = ynorm
    elif alg.d < 6:
        num, denom = hitzer_inv(y, symbolic=True)
    else:
        num, denom = shirokov_inv(y, symbolic=True)

    if symbolic:
        return Fraction(num, denom)

    d = denom.e
    return num.map(lambda v: v / d)


def hitzer_inv(x: "MultiVector", symbolic: bool = False) -> "MultiVector":
    """
    Generate code for the inverse of :code:`x` using the Hitzer inverse,
    which works up to 5D algebras.
    """
    alg = x.algebra
    d = alg.d
    two = alg.multivector(e=2)
    if d == 0:
        num = alg.blades.e
    elif d == 1:
        num = involute(x)
    elif d == 2:
        num = conjugate(x)
    elif d == 3:
        xconj = conjugate(x)
        num = gp(xconj, reverse(gp(x, xconj)))
    elif d == 4:
        xconj = conjugate(x)
        x_xconj = gp(x, xconj)
        num = gp(xconj, sub(x_xconj, gp(two, grade(x_xconj, 3, 4))))
    elif d == 5:
        xconj = conjugate(x)
        x_xconj = gp(x, xconj)
        combo = gp(xconj, reverse(x_xconj))
        x_combo = gp(x, combo)
        num = gp(combo, sub(x_combo, gp(two, grade(x_combo, 1, 4))))
    else:
        raise NotImplementedError(f"Closed form inverses are not known in {d=} dimensions.")
    denom = sp(x, num)

    if symbolic:
        return Fraction(num, denom)
    denom = denom.e
    return num.map(lambda v: v / denom)


def shirokov_inv(x: "MultiVector", symbolic: bool = False) -> "MultiVector":
    """
    Generate code for the inverse of :code:`x` using the Shirokov inverse,
    which is works in any algebra, but it can be expensive to compute.
    """
    alg = x.algebra
    n = 2 ** ((alg.d + 1) // 2)
    supply = power_supply(x, tuple(range(1, n + 1)), operation=gp)  # Generate powers of x efficiently.
    powers = []
    cs = []
    xs = []
    for i in range(1, n + 1):
        powers.append(next(supply))
        xi = powers[i - 1]
        for j in range(i - 1):
            power_idx = i - j - 2
            xi_diff = gp(powers[power_idx], cs[j])
            xi = sub(xi, xi_diff)
        if xi.grades == (0,):
            break
        xs.append(xi)
        cs.append(s if (s := xi.e) == 0 else n * s / i)

    if i == 1:
        adj = alg.blades.e
    else:
        adj = sub(xs[-1], cs[-1])

    if symbolic:
        return Fraction(adj, xi)
    xi = xi.e
    return adj.map(lambda v: v / xi)


def div(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    """
    Generate code for :math:`x y^{-1}`.
    """
    num, denom = inv(y, symbolic=True)
    if not denom:
        raise ZeroDivisionError
    d = denom.e
    return gp(x, num).map(lambda v: v / d)


def normsq(x: "MultiVector") -> "MultiVector":
    return gp(x, reverse(x))


def outerexp(x: "MultiVector", asterms: bool = False) -> "MultiVector":
    alg = x.algebra
    if len(x.grades) != 1:
        warnings.warn('Outer exponential might not converge for mixed-grade multivectors.', RuntimeWarning)
    k = alg.d

    Ws = [alg.scalar(e=1), x]
    j = 2
    while j <= k:
        Wj = op(Ws[-1], x)
        # Dividing like this avoids floating point numbers, which is excellent.
        jinv = PyFraction(1, j)
        Wj._values = [v*jinv for v in Wj._values]
        if Wj:
            Ws.append(Wj)
            j += 1
        else:
            break

    if asterms:
        return Ws
    return reduce(operator.add, Ws)


def outersin(x: "MultiVector") -> "MultiVector":
    odd_Ws = outerexp(x, asterms=True)[1::2]
    outersin = reduce(operator.add, odd_Ws)
    return outersin


def outercos(x: "MultiVector") -> "MultiVector":
    even_Ws = outerexp(x, asterms=True)[0::2]
    outercos = reduce(operator.add, even_Ws)
    return outercos


def outertan(x: "MultiVector") -> "MultiVector":
    Ws = outerexp(x, asterms=True)
    even_Ws, odd_Ws = Ws[0::2], Ws[1::2]
    outercos = reduce(operator.add, even_Ws)
    outersin = reduce(operator.add, odd_Ws)
    outertan = div(outersin, outercos)
    return outertan


def add(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] + v
        else:
            vals[k] = v
    return dict_to_multivector(vals, x.algebra)


def sub(x: "MultiVector", y: "MultiVector") -> "MultiVector":
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] - v
        else:
            vals[k] = -v
    return dict_to_multivector(vals, x.algebra)


def neg(x: "MultiVector") -> "MultiVector":
    return dict_to_multivector({k: -v for k, v in x.items()}, x.algebra)


def involutions(x: "MultiVector", invert_grades: tuple[int, int] = (2, 3)) -> "MultiVector":
    """
    Codegen for the involutions of Clifford algebras:
    reverse, grade involute, and Clifford involution.

    :param invert_grades: The grades that flip sign under this involution mod 4, e.g. (2, 3) for reversion.
    """
    res = {k: -v if bin(k).count('1') % 4 in invert_grades else v
           for k, v in x.items()}
    return dict_to_multivector(res, x.algebra)


def reverse(x: "MultiVector") -> "MultiVector":
    return involutions(x, invert_grades=(2, 3))


def involute(x: "MultiVector") -> "MultiVector":
    return involutions(x, invert_grades=(1, 3))


def conjugate(x: "MultiVector") -> "MultiVector":
    return involutions(x, invert_grades=(1, 2))


def sqrt(x: "MultiVector") -> "MultiVector":
    """
    Take the square root using the study number approach as described in
    https://doi.org/10.1002/mma.8639
    """
    alg = x.algebra
    if x.grades == (0,):
        return x.map(lambda v: v**0.5)
    a, bI = grade(x, 0), sub(x, grade(x, 0))
    has_solution = len(x.grades) <= 2 and 0 in x.grades
    if not has_solution:
        warnings.warn("Cannot verify that we really are taking the sqrt of a Study number.", RuntimeWarning)

    bI_sq = gp(bI, bI)
    if not bI_sq:
        cp = a.e**0.5
    else:
        normS = (sub(gp(a, a), bI_sq)).e
        cp = (0.5 * (a.e + normS**0.5))**0.5
    cp = alg.multivector(e=cp)
    return add(div(gp(alg.multivector(e=0.5), bI), cp), cp)


def polarity(x: "MultiVector", undual: bool = False) -> "MultiVector":
    pss = x.algebra.multivector({len(x.algebra) - 1: 1})
    if undual:
        return gp(x, pss)
    key_pss = len(x.algebra) - 1
    sign = x.algebra.signs[key_pss, key_pss]
    if sign == -1:
        return - gp(x, pss)
    return gp(x, pss)


def unpolarity(x: "MultiVector") -> "MultiVector":
    return polarity(x, undual=True)


def hodge(x: "MultiVector", undual: bool = False) -> "MultiVector":
    if undual:
        res = {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[key_dual, eI] < 0 else v
               for eI, v in x.items()}
    else:
        res = {(key_dual := len(x.algebra) - 1 - eI): -v if x.algebra.signs[eI, key_dual] < 0 else v
               for eI, v in x.items()}
    return dict_to_multivector(res, x.algebra)


def unhodge(x: "MultiVector") -> "MultiVector":
    return hodge(x, undual=True)
