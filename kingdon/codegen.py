from __future__ import annotations

import string
from itertools import product, combinations, groupby
from collections import namedtuple
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
from sympy.printing.numpy import NumPyPrinter


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


def codegen_product(*mvs, filter_func=None, sign_func=None, keyout_func=operator.xor, symbolic=False):
    """
    Helper function for the codegen of all product-type functions.

    :param mvs: Positional-argument :class:`~kingdon.multivector.MultiVector`.
    :param filter_func: A condition which should be true in the preprocessing of terms.
        Input is a TermTuple.
    :param sign_func: function to compute sign between terms. E.g. algebra.signs[ei, ej]
        for metric dependent products. Input: 2-tuple of blade indices, e.g. (ei, ej).
    :param asdict: If true, return the dict of strings before converting to a function.
    """
    sortfunc = lambda x: x.key_out
    algebra = mvs[0].algebra
    if sign_func is None:
        sign_func = lambda pair: algebra.signs[pair]

    # If sign == 0, then the term should be disregarded since it is zero
    terms = filter(lambda tt: tt.sign, (term_tuple(items, sign_func, keyout_func=keyout_func)
                                        for items in product(*(mv.items() for mv in mvs))))
    if filter_func is not None:
        terms = filter(filter_func, terms)
    # TODO: Can we loop over the basis blades in such a way that no sort is needed?
    sorted_terms = sorted(terms, key=sortfunc)
    if not symbolic:
        return {k: "".join(term.termstr for term in group)
               for k, group in groupby(sorted_terms, key=sortfunc)}
    else:
        res = {k: Add(*(Mul(*(term.values_in if term.sign == 1 else (term.sign, *term.values_in)), evaluate=False)
                        for term in group), evaluate=False)
               for k, group in groupby(sorted_terms, key=sortfunc)}
        return algebra.multivector(res)


def codegen_gp(x, y, symbolic=False):
    """
    Generate the geometric product between :code:`x` and :code:`y`.

    :param x: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param y: Fully symbolic :class:`~kingdon.multivector.MultiVector`.
    :param symbolic: If true, return a symbolic :class:`~kingdon.multivector.MultiVector` instead of a lambda function.
    :return: tuple with integers indicating the basis blades present in the
        product in binary convention, and a lambda function that perform the product.
    """
    return codegen_product(x, y, symbolic=symbolic)


def codegen_sw(x, y):
    """
    Generate the projection of :code:`x` onto :code:`y`: :math:`x y \widetilde{x}`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    if x.algebra.simplify:
        res = codegen_product(x, y, ~x, symbolic=True)
        res = {k: str(simp_expr) for k, expr in res.items() if (simp_expr := expand(expr))}
    else:
        res = codegen_product(x, y, ~x, symbolic=False)
    return res


def codegen_cp(x, y, symbolic=False):
    """
    Generate the commutator product of :code:`x` and :code:`y`: :code:`x.cp(y) = 0.5*(x*y-y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] - algebra.signs[tt.keys_in[::-1]])
    return codegen_product(x, y, filter_func=filter_func, symbolic=symbolic)


def codegen_acp(x, y, symbolic=False):
    """
    Generate the anti-commutator product of :code:`x` and :code:`y`: :code:`x.acp(y) = 0.5*(x*y+y*x)`.

    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: (algebra.signs[tt.keys_in] + algebra.signs[tt.keys_in[::-1]])
    return codegen_product(x, y, filter_func=filter_func, symbolic=symbolic)


def codegen_ip(x, y, diff_func=abs, symbolic=False):
    """
    Generate the inner product of :code:`x` and :code:`y`.

    :param diff_func: How to treat the difference between the binary reps of the basis blades.
        if :code:`abs`, compute the symmetric inner product. When :code:`lambda x: -x` this
        function generates left-contraction, and when :code:`lambda x: x`, right-contraction.
    :return: tuple of keys in binary representation and a lambda function.
    """
    algebra = x.algebra
    filter_func = lambda tt: tt.key_out == diff_func(tt.keys_in[0] - tt.keys_in[1])
    return codegen_product(x, y, filter_func=filter_func, symbolic=symbolic)


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
    filter_func = lambda tt: tt.key_out == abs(tt.keys_in[0] - tt.keys_in[1]) ^ tt.keys_in[2]
    if x.algebra.simplify:
        res = codegen_product(x, y, ~y, filter_func=filter_func, symbolic=True)
        return {k: str(simp_expr) for k, expr in res.items() if (simp_expr := expand(expr))}
    return codegen_product(x, y, ~y, filter_func=filter_func, symbolic=False)


def codegen_op(x, y, symbolic=False):
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
    return codegen_product(x, y, filter_func=filter_func, sign_func=sign_func, symbolic=symbolic)


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

def codegen_inv(y, x=None, symbolic=False):
    alg = y.algebra
    # As preprocessing we invert y*~y since y^{-1} = ~y / (y*~y)
    if len(y) == 1 and y.grades == (0,):
        adj_y, denom_y = x or 1, y.e
    else:
        ynormsq = y.normsq()

        if ynormsq.grades == tuple():
            raise ZeroDivisionError
        elif ynormsq.grades == (0,):
            adj_y, denom_y = ~y if x is None else x * ~y, ynormsq.e
        else:
            # Make a mv with the same components as ynormsq, and invert that instead.
            # Although this requires some more bookkeeping, it is much more performant.
            z = alg.multivector(name='z', keys=ynormsq.keys())
            adj_z, denom_z = codegen_shirokov_inv(z, symbolic=True)

            # Same trick, make a mv that mimicks adj_z and multiply it with ~y.
            A_z = alg.multivector(name='A_z', keys=adj_z.keys())
            # Faster to this `manually` instead of with ~a * A_z
            res = codegen_product(~y, A_z, symbolic=True)
            A_y = res if x is None else x * res

            # Replace all the dummy A_z symbols by the expressions in adj_z.
            subs_dict = dict(zip(A_z.values(), adj_z.values()))
            adj_y = A_y.subs(subs_dict)
            # Replace all the dummy b symbols by the expressions in anormsq to
            # recover the actual adjoint of a and corresponding denominator.
            subs_dict = dict(zip(z.values(), ynormsq.values()))
            denom_y = denom_z.subs(subs_dict)
            adj_y = adj_y.subs(subs_dict)

    if symbolic:
        return Fraction(adj_y, denom_y)

    d = alg.scalar(name='d')
    denom_y_inv = alg.scalar([1 / denom_y])
    yinv = alg.multivector({k: Mul(d[0], v, evaluate=False) for k, v in adj_y.items()})

    # Prepare all the input for lambdify
    args = {'y': y.values()}
    expr = yinv.values()
    dependencies = [(d.values(), denom_y_inv.values())]
    return CodegenOutput(
        yinv.keys(),
        lambdify(args, expr, funcname=f'inv_{y:keys_binary}', dependencies=dependencies, cse=alg.cse)
    )


def codegen_shirokov_inv(x, symbolic=False):
    """
    Generate code for the inverse of :code:`x` using the Shirokov inverse,
    which is works in any algebra, but it can be expensive to compute.
    """
    alg = x.algebra
    n = sympify(2 ** ((alg.d + 1) // 2))  # Sympify ratio to keep the ratios exact and avoid floating point errors.
    supply = power_supply(x, tuple(range(1, n + 1)))
    powers = []
    cs = []
    xs = []
    for i in range(1, n + 1):
        powers.append(next(supply))
        xi = powers[i - 1]
        for j in range(i - 1):
            power_idx = i - j - 2
            xi_diff = x.fromkeysvalues(
                alg, keys=xi.keys(),
                values=tuple(Mul(cs[j], v) for v in powers[power_idx].values())
            )
            xi = xi - xi_diff
            xi = alg.multivector(
                {k: simp_expr for k, expr in xi.items() if (simp_expr := expand(expr))}
            )
        if xi.grades == (0,):
            break
        xs.append(xi)
        cs.append((n / i) * xi[0])

    if i == 1:
        adj = alg.scalar([1])
    else:
        adj = xs[-1] - cs[-1]

    if symbolic:
        return Fraction(adj, xi[0])
    return alg.multivector({k: v / xi[0] for k, v in adj.items()})


def codegen_div(x, y):
    """
    Generate code for :math:`x y^{-1}`.
    """
    alg = x.algebra
    num, denom = codegen_inv(y, x, symbolic=True)
    if not denom:
        raise ZeroDivisionError
    d = alg.scalar(name='d')
    denom_inv = alg.scalar([1 / denom])
    res = alg.multivector({k: Mul(d[0], v, evaluate=False) for k, v in num.items()})

    # Prepare all the input for lambdify
    args = {'x': x.values(), 'y': y.values()}
    expr = res.values()
    dependencies = [(d.values(), denom_inv.values())]
    return CodegenOutput(
        res.keys(),
        lambdify(args, expr, funcname=f'inv_{x:keys_binary}_x_{y:keys_binary}',
                 dependencies=dependencies, cse=alg.cse)
    )


def codegen_normsq(x):
    if x.algebra.simplify:
        res = codegen_product(x, ~x, symbolic=True)
        res = {k: simp_expr for k, expr in res.items() if (simp_expr := expand(expr))}
    else:
        res = codegen_product(x, ~x, symbolic=False)
    return res


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


def _lambdify_mv(free_symbols, mv):
    func = lambdify(free_symbols, mv.values(), funcname=f'custom_{mv:keys_binary}', cse=mv.algebra.cse)
    return CodegenOutput(tuple(mv.keys()), func)


def do_codegen(codegen, *mvs) -> CodegenOutput:
    """
    :param codegen: callable that performs codegen for the given :code:`mvs`. This can be any callable
        that returns either a :class:`~kingdon.multivector.MultiVector`, a dictionary, or an instance of :class:`CodegenOutput`.
    :param mvs: Any remaining positional arguments are taken to be symbolic :class:`~kingdon.multivector.MultiVector`'s.
    :return: Instance of :class:`CodegenOutput`.
    """
    res = codegen(*mvs)
    if isinstance(res, CodegenOutput):
        return res
    funcname = f'{codegen.__name__}_' + '_x_'.join(f"{mv:keys_binary}" for mv in mvs)
    args = {arg_name: arg.values() for arg_name, arg in zip(string.ascii_uppercase, mvs)}
    exprs = tuple(res.values())
    return CodegenOutput(
        tuple(res.keys()),
        lambdify(args, exprs, funcname=funcname, cse=mvs[0].algebra.cse)
    )


def lambdify(args: dict, exprs: tuple, funcname: str, dependencies: tuple = None, printer=NumPyPrinter, dummify=False, cse=False):
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
    if printer is NumPyPrinter:
        printer = NumPyPrinter(
            {'fully_qualified_modules': False, 'inline': True,
             'allow_unknown_functions': True,
             'user_functions': {}}
        )

    names = tuple(arg if isinstance(arg, str) else arg.name for arg in args.keys())
    iterable_args = tuple(args.values())

    funcprinter = KingdonPrinter(printer, dummify)

    # TODO: Extend CSE to include the dependencies.
    lhsides, rhsides = zip(*dependencies) if dependencies else ([], [])
    if cse and not any(isinstance(expr, str) for expr in exprs):
        if not callable(cse):
            from sympy.simplify.cse_main import cse
        if dependencies:
            all_exprs = (*exprs, *rhsides)
            cses, _all_exprs = cse(all_exprs, list=False)
            _exprs, _rhsides = _all_exprs[:-len(rhsides)], _all_exprs[len(exprs):]
            cses.extend(tuple(zip(flatten(lhsides), flatten(_rhsides))))
        else:
            cses, _exprs = cse(exprs, list=False)
    else:
        cses, _exprs = list(zip(flatten(lhsides), flatten(rhsides))), exprs

    if not any(_exprs):
        _exprs = tuple('0' for expr in _exprs)
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
            elif arg.is_symbol:
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
