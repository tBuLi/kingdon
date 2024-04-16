from __future__ import annotations

import string
from itertools import product, combinations, groupby
from collections import namedtuple, defaultdict
from typing import NamedTuple, Callable, Tuple, Dict
from functools import reduce, cached_property
import linecache
import warnings
import operator
from dataclasses import dataclass
import inspect
import builtins
import keyword

from sympy import simplify, sympify, Add, Mul, Symbol, expand
from sympy.utilities.iterables import iterable, flatten
from sympy.printing.lambdarepr import LambdaPrinter


@dataclass
class AdditionChains:
    limit: int

    @cached_property
    def minimal_chains(self) -> Dict[int, Tuple[int, ...]]:
        chains = {1: (1,)}
        while any(i not in chains for i in range(1, self.limit + 1)):
            for chain in chains.copy().values():
                right_summand = chain[-1]
                for left_summand in chain:
                    value = left_summand + right_summand
                    if value <= self.limit and value not in chains:
                        chains[value] = (*chain, value)
        return chains

    def __getitem__(self, n: int) -> Tuple[int, ...]:
        return self.minimal_chains[n]

    def __contains__(self, item):
        return self[item]

def power_supply(x: "MultiVector", exponents: Tuple[int, ...], operation: Callable[["MultiVector", "MultiVector"], "MultiVector"] = operator.mul):
    """
    Generates powers of a given multivector using the least amount of multiplications.
    For example, to raise a multivector :math:`x` to the power :math:`a = 15`, only 5
    multiplications are needed since :math:`x^{2} = x * x`, :math:`x^{3} = x * x^2`,
    :math:`x^{5} = x^2 * x^3`, :math:`x^{10} = x^5 * x^5`, :math:`x^{15} = x^5 * x^{10}`.
    The :class:`power_supply` uses :class:`AdditionChains` to determine these shortest
    chains.

    When called with only a single integer, e.g. :code:`power_supply(x, 15)`, iterating
    over it yields the above sequence in order; ending with :math:`x^{15}`.

    When called with a sequence of integers, the generator instead returns only the requested terms.


    :param x: The MultiVector to be raised to a power.
    :param exponents: When an :code:`int`, this generates the shortest possible way to
        get to :math:`x^a`, where :math:`x`
    """
    if isinstance(exponents, int):
        target = exponents
        addition_chains = AdditionChains(target)
        exponents = addition_chains[target]
    else:
        addition_chains = AdditionChains(max(exponents))

    powers = {1: x}
    for step in exponents:
        if step not in powers:
            chain = addition_chains[step]
            powers[step] = operation(powers[chain[-2]], powers[step - chain[-2]])

        yield powers[step]


class TermTuple(NamedTuple):
    """
    TermTuple represents a single monomial in a product of multivectors.

    :param key_out: is the basis blade to which this monomial belongs.
    :param keys_in: are the input basis blades in this monomial.
    :param sign: Sign of the monomial.
    :param values_in: Input values. Typically, tuple of :class:`sympy.core.symbol.Symbol`.
    :param termstr: The string representation of this monomial, e.g. '-x*y'.
    """
    key_out: int
    keys_in: Tuple[int]
    sign: int
    values_in: Tuple["sympy.core.symbol.Symbol"]
    termstr: str


class CodegenOutput(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    """
    keys_out: Tuple[int]
    func: Callable


def term_tuple(items, sign_func, keyout_func=operator.xor):
    """
    Create a single term in a multivector product between the basis blades present in `items`.
    """
    keys_in, values_in = zip(*items)
    sign = sign_func(keys_in)
    if not sign:
        return TermTuple(key_out=0, keys_in=keys_in, sign=sign, values_in=values_in, termstr='')
    key_out = keyout_func(*keys_in)
    return TermTuple(key_out, keys_in, sign, values_in,
                     f'{"+" if sign > 0 else "-"}{"*".join(str(v) for v in values_in)}')


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
    algebra = x.algebra
    if sign_func is None:
        sign_func = lambda pair: algebra.signs[pair]

    # If sign == 0, then the term should be disregarded since it is zero
    terms = filter(lambda tt: tt.sign, (term_tuple(items, sign_func, keyout_func=keyout_func)
                                        for items in product(x.items(), y.items())))
    if filter_func is not None:
        terms = filter(filter_func, terms)

    res = defaultdict(str)
    for term in terms:
        if term.key_out in res:
            res[term.key_out] += term.termstr
        else:
            res[term.key_out] = term.termstr[1:] if term.termstr[0] == '+' else term.termstr
    return dict(sorted(res.items()))


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
    """
    Generate the conjugation of :code:`y` by :code:`x`: :math:`x y \widetilde{x}`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return x * y * ~x


def codegen_cp(x, y):
    """
    Generate the commutator product of :code:`x` and :code:`y`: :code:`x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] - algebra.signs[tt.keys_in[::-1]])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_acp(x, y):
    """
    Generate the anti-commutator product of :code:`x` and :code:`y`: :code:`x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] + algebra.signs[tt.keys_in[::-1]])
    return codegen_product(x, y, filter_func=filter_func)


def codegen_ip(x, y, diff_func=abs):
    """
    Generate the inner product of :code:`x` and :code:`y`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: tt.key_out == diff_func(tt.keys_in[0] - tt.keys_in[1])
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
    """
    Generate the projection of :code:`x` onto :code:`y`: :math:`(x \cdot y) \widetilde{y}`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    return (x | y) * ~y


def codegen_op(x, y):
    """
    Generate the outer product of :code:`x` and :code:`y`: :code:`x.op(y) = x ^ y`.

    :x: MultiVector
    :y: MultiVector
    :return: dictionary with integer keys indicating the corresponding basis blade in binary convention,
        and values which are a 3-tuple of indices in `x`, indices in `y`, and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: tt.key_out == sum(tt.keys_in)
    sign_func = lambda pair: (-1)**algebra.swaps[pair]
    return codegen_product(x, y, filter_func=filter_func, sign_func=sign_func)


def codegen_rp(x, y):
    """
    Generate the regressive product of :code:`x` and :code:`y`:,
    :math:`x \\vee y`.

    :param x:
    :param y:
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
    )


Fraction = namedtuple('Fraction', ['numer', 'denom'])
Fraction.__doc__ = """
Tuple representing a fraction.
"""


class LambdifyInput(NamedTuple):
    """ Strike package for the Lambdify function. """
    funcname: str
    args: dict
    expr_dict: dict
    dependencies: list


def codegen_inv(y, x=None, symbolic=False):
    alg = y.algebra
    if alg.d < 6:
        num, denom = codegen_hitzer_inv(y, symbolic=True)
    else:
        num, denom = codegen_shirokov_inv(y, symbolic=True)
    num = num if x is None else x * num

    if symbolic:
        return Fraction(num, denom)

    d = alg.scalar(name='d', symbolcls=alg.codegen_symbolcls)
    denom_inv = alg.scalar([1 / denom])
    yinv = num * d.e  #TODO: this multiply is too gready, would be better if it didnt distribute, & reinstate CSE

    # Prepare all the input for lambdify
    args = {'y': y.values()}
    expr_dict = dict(yinv.items())
    dependencies = list(zip(d.values(), denom_inv.values()))
    return LambdifyInput(
        funcname=f'codegen_inv_{y.type_number}',
        expr_dict=expr_dict,
        args=args,
        dependencies=dependencies,
    )

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
    denom = (x * num).e

    if symbolic:
        return Fraction(num, denom)
    return alg.multivector({k: v / denom for k, v in num.items()})

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
        return Fraction(adj, xi.e)
    return alg.multivector({k: v / xi.e for k, v in adj.items()})


def codegen_div(x, y):
    """
    Generate code for :math:`x y^{-1}`.
    """
    alg = x.algebra
    num, denom = codegen_inv(y, x, symbolic=True)
    if not denom:
        raise ZeroDivisionError
    d = alg.scalar(name='d', symbolcls=alg.codegen_symbolcls)
    denom_inv = alg.scalar([1 / denom])
    res = num * d.e

    # Prepare all the input for lambdify
    args = {'x': x.values(), 'y': y.values()}
    expr_dict = dict(res.items())
    dependencies = list(zip(d.values(), denom_inv.values()))
    return LambdifyInput(
        funcname=f'div_{x.type_number}_x_{y.type_number}',
        expr_dict=expr_dict,
        args=args,
        dependencies=dependencies,
    )


def codegen_normsq(x):
    return x * ~x


def codegen_outerexp(x, asterms=False):
    alg = x.algebra
    if len(x.grades) != 1:
        warnings.warn('Outer exponential might not converge for mixed-grade multivectors.', RuntimeWarning)
    k = alg.d

    Ws = [alg.scalar([1]), x]
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
    return vals


def codegen_sub(x, y):
    vals = dict(x.items())
    for k, v in y.items():
        if k in vals:
            vals[k] = vals[k] - v
        else:
            vals[k] = -v
    return vals

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
        return {0: f'({str(x.e)}**0.5)'}
    a, bI = x.grade(0), x - x.grade(0)
    has_solution = len(x.grades) <= 2 and 0 in x.grades
    if not has_solution:
        warnings.warn("Cannot verify that we really are taking the sqrt of a Study number.", RuntimeWarning)

    bI_sq = bI * bI
    if not bI_sq:
        cp = f'({str(a.e)}**0.5)'
    else:
        normS = (a * a - bI * bI).e
        cp = f'(0.5 * ({str(a.e)} + {str(normS)}**0.5)) ** 0.5'
    c = alg.scalar(name='c')
    c2_inv = alg.scalar(name='c2_inv')
    dI = bI * c2_inv
    res = c + dI

    # Prepare all the input for lambdify
    args = {'x': x.values()}
    expr_dict = dict(res.items())
    dependencies = [*zip(c.values(), [cp]), *zip(c2_inv.values(), [f'0.5 / {cp}'])]
    return LambdifyInput(
        funcname=f'sqrt_{x.type_number}',
        expr_dict=expr_dict,
        args=args,
        dependencies=dependencies,
    )


def codegen_polarity(x, undual=False):
    if undual:
        return x * x.algebra.pss
    sign = x.algebra.signs[-1, -1]
    if sign == 1:
        return x * x.algebra.pss
    if sign == -1:
        return x * (-x.algebra.pss)
    if sign == 0:
        raise ZeroDivisionError


def codegen_unpolarity(x):
    return codegen_polarity(x, undual=True)


def codegen_hodge(x, undual=False):
    if undual:
        return x.algebra.multivector(
            {len(x.algebra) - 1 - eI: x.algebra.signs[len(x.algebra) - 1 - eI, eI] * val
             for eI, val in x.items()}
        )
    return x.algebra.multivector(
        {len(x.algebra) - 1 - eI: x.algebra.signs[eI, len(x.algebra) - 1 - eI] * val
         for eI, val in x.items()}
    )


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


def do_codegen(codegen, *mvs) -> CodegenOutput:
    """
    :param codegen: callable that performs codegen for the given :code:`mvs`. This can be any callable
        that returns either a :class:`~kingdon.multivector.MultiVector`, a dictionary, or an instance of :class:`CodegenOutput`.
    :param mvs: Any remaining positional arguments are taken to be symbolic :class:`~kingdon.multivector.MultiVector`'s.
    :return: Instance of :class:`CodegenOutput`.
    """
    algebra = mvs[0].algebra
    namespace = algebra.numspace

    res = codegen(*mvs)

    if isinstance(res, CodegenOutput):
        return res

    if isinstance(res, LambdifyInput):
        funcname = res.funcname
        args = res.args
        dependencies = res.dependencies
        res = res.expr_dict
    else:
        funcname = f'{codegen.__name__}_' + '_x_'.join(f"{mv.type_number}" for mv in mvs)
        args = {arg_name: arg.values() for arg_name, arg in zip(string.ascii_uppercase, mvs)}
        dependencies = None

    # Sort the keys in canonical order
    res = {bin: res[bin] if isinstance(res, dict) else getattr(res, canon)
           for canon, bin in algebra.canon2bin.items() if bin in res.keys()}

    if not algebra.cse and any(isinstance(v, str) for v in res.values()):
        return func_builder(res, *mvs, funcname=funcname)


    keys, exprs = tuple(res.keys()), list(res.values())
    func = lambdify(args, exprs, funcname=funcname, cse=algebra.cse, dependencies=dependencies)
    return CodegenOutput(
        keys, func
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


def func_builder(res_vals: defaultdict, *mvs, funcname: str) -> CodegenOutput:
    """
    Build a Python function for the product between given multivectors.

    :param res_vals: Dict to be converted into a function. The keys correspond to the basis blades in binary,
        while the values are strings to be converted into source code.
    :param mvs: all the multivectors that the resulting function is a product of.
    :param funcname: Name of the function. Be aware: if a function by that name already existed, it will be overwritten.
    :return: tuple of output keys of the callable, and the callable.
    """
    args = string.ascii_uppercase[:len(mvs)]
    header = f'def {funcname}({", ".join(args)}):'
    if res_vals:
        body = ''
        for mv, arg in zip(mvs, args):
            body += f'    [{", ".join(str(v) for v in mv.values())}] = {arg}\n'
        return_val = f'    return [{", ".join(res_vals.values())},]'
    else:
        body = ''
        return_val = f'    return list()'
    func_source = f'{header}\n{body}\n{return_val}'

    # Dynamically build a function
    func_locals = {}
    c = compile(func_source, funcname, 'exec')
    exec(c, {}, func_locals)

    # Add the generated code to linecache such that it is inspect-safe.
    linecache.cache[funcname] = (len(func_source), None, func_source.splitlines(True), funcname)
    func = func_locals[funcname]
    return CodegenOutput(tuple(res_vals.keys()), func)


def lambdify(args: dict, exprs: list, funcname: str, dependencies: tuple = None, printer=LambdaPrinter, dummify=False, cse=False):
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

    It is recommended not to call this function directly, but rather to use
    :func:`do_codegen` which provides a clean API around this function.

    :param args: dictionary of type dict[str | Symbol, tuple[Symbol]].
    :param exprs: tuple[Expr]
    :param funcname: string to be used as the bases for the name of the function.
    :param dependencies: These are extra expressions that can be provided such that quantities can be precomputed.
        For example, in the inverse of a multivector, this is used to compute the scalar denominator only once,
        after which all values in expr are multiplied by it. When :code:`cse = True`, these dependencies are also
        included in the CSE process.
    :param cse: If :code:`True` (default), CSE is applied to the expressions and dependencies.
        This typically greatly improves performance and reduces numba's initialization time.
    :return: Function that represents that can be used to calculate the values of exprs.
    """
    if printer is LambdaPrinter:
        printer = LambdaPrinter(
            {'fully_qualified_modules': False, 'inline': True,
             'allow_unknown_functions': True,
             'user_functions': {}}
        )

    tosympy = lambda x: x.tosympy() if hasattr(x, 'tosympy') else x
    args = {name: [tosympy(v) for v in values]
            for name, values in args.items()}
    exprs = [tosympy(expr) for expr in exprs]
    if dependencies is not None:
        dependencies = [(tosympy(y), tosympy(x)) for y, x in dependencies]
    names = tuple(arg if isinstance(arg, str) else arg.name for arg in args.keys())
    iterable_args = tuple(args.values())

    funcprinter = KingdonPrinter(printer, dummify)

    # TODO: Extend CSE to include the dependencies.
    lhsides, rhsides = zip(*dependencies) if dependencies else ([], [])
    if cse and not any(isinstance(expr, str) for expr in exprs):
        if not callable(cse):
            from sympy.simplify.cse_main import cse
        if dependencies:
            all_exprs = [*exprs, *rhsides]
            cses, _all_exprs = cse(all_exprs, list=False, order='none', ignore=lhsides)
            _exprs, _rhsides = _all_exprs[:-len(rhsides)], _all_exprs[len(exprs):]
            cses.extend(list(zip(flatten(lhsides), flatten(_rhsides))))
        else:
            cses, _exprs = cse(exprs, list=False)
    else:
        cses, _exprs = list(zip(flatten(lhsides), flatten(rhsides))), exprs

    if not any(_exprs):
        _exprs = list('0' for expr in _exprs)
    funcstr = funcprinter.doprint(funcname, iterable_args, names, _exprs, cses=cses)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace = {'builtins': builtins, 'range': range}

    funclocals = {}
    filename = f'<{funcname}>'
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    func = funclocals[funcname]
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

    def doprint(self, funcname, args, names, expr, *, cses=()):
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

        for name, argstr in zip(names, argstrs):
            if iterable(argstr):
                funcargs.append(name)
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
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

        str_expr = _recursive_to_string(self._exprrepr, expr)

        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)
        funcbody.append('return {}'.format(str_expr))

        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])

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
            return '[{}]'.format(', '.join(
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
