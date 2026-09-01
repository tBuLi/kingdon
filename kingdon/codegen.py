from __future__ import annotations

import re
import string
from itertools import chain
from typing import NamedTuple, Callable, Tuple, Dict, Optional, List
import linecache
import inspect
import builtins
import keyword
import copy

from sympy.utilities.iterables import iterable, flatten
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.simplify.cse_main import numbered_symbols
from sympy import Symbol, sympify

from kingdon.polynomial import poly_cse, poly_format, Polynomial, RationalPolynomial
from kingdon.multivector import MultiVector, MultiVectorType


class CompiledExpression(NamedTuple):
    """
    Output of a codegen function.

    :param keys_out: tuple with the output blades in binary rep.
    :param func: callable that takes (several) sequence(s) of values
        returns a tuple of :code:`len(keys_out)`.
    :param wrapped_func: decorated func if a wrapper was provided, else identical to func.
    :param mvtype: type of the output multivector. Defaults to :code:`MultiVector`.
    """
    algebra: "Algebra"
    keys_out: Tuple[int]
    func: Callable
    mvtype: MultiVectorType = MultiVector
    output_mv_idx: int | None = None
    wrapped_func: Callable | None = None
    values_asarray: Callable | None = None

    def __call__(self, *mvs):
        issymbolic = any(mv.issymbolic for mv in mvs)
        values_in = tuple(mv.values() for mv in mvs)
        values_out = self.func(*values_in) if issymbolic else self.wrapped_func(*values_in)
        if self.output_mv_idx is not None: return None  # The function uses .set
        return self.mvtype.fromkeysvalues(
            self.algebra, self.keys_out, values_out, values_asarray=self.values_asarray, raw=issymbolic
        )


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

    best_MVType, best_layout, best_cost = MultiVector, {}, None
    for cls, L in layouts.items():
        if MVType is not None and not issubclass(cls, MVType):
            continue
        free = {k for k, v in L.items() if v is Ellipsis}
        fixed_items = {(k, v) for k, v in L.items() if v is not Ellipsis}
        all_keys = list(L.keys())
        if not res_free.issubset(free):
            continue
        if not fixed_items.issubset(res_fixed_items):
            continue
        if not res_fixed_keys.issubset(all_keys):
            continue
        cost = (len(free & res_fixed_keys), len(free - res_keys))
        if best_cost is None or cost < best_cost:
            best_MVType, best_layout, best_cost = cls, L, cost
            if cost == (0, 0):
                break  # perfect fit; layouts are iterated in insertion order so this is optimal

    return best_MVType, best_layout


def do_compile_symbolic(codegen, *mvs, printer=None, func_printer=None, wrapper=None, values_asarray=None) -> CompiledExpression:
    """
    :param codegen: callable that performs codegen for the given :code:`mvs`. This can be any callable
        that returns a :class:`~kingdon.multivector.MultiVector`.
    :param mvs: Any remaining positional arguments are taken to be symbolic :class:`~kingdon.multivector.MultiVector`'s.
    :param printer: The sympy style printer used to generate the code with sympy-style printing.
    :param func_printer: The sympy style evaluator printer used to generate the code with sympy-style printing.
    :return: Instance of :class:`CompiledExpression`.
    """
    algebra = mvs[0].algebra
    mvs_orig = [copy.deepcopy(mv) for mv in mvs]

    res = codegen(*(mv.asmvtype() for mv in mvs))

    MVType = MultiVector
    output_mv_idx = None  # If codegen modified one of the mvs using set, this will be the index of the modified mv.
    if res is None:
        output_mv_idx = next(i for i, mv in enumerate(mvs) if mv != mvs_orig[i])
        res = mvs[output_mv_idx]
        mvs = mvs_orig
    else:
        def is_number(x):
            try: float(x); return True
            except (ValueError, TypeError): return False
        res_layout = {k: float(f) if is_number(f := str(v)) else ... for k, v in res.items()}
        MVType, layout = resolve_layout(algebra._type_layouts, res_layout)

        if layout:
            res = dict(res.items())
            res = {k: res[k] for k, v in layout.items() if v == ... and k in res}

    funcname = f'{codegen.__name__}_' + '_x_'.join(f"{format(mv[0].type_number if isinstance(mv, list) else mv.type_number, 'X')}" for mv in mvs)
    args = {arg_name: [tuple(chain(*(x.values() for x in arg)))] if isinstance(arg, list) else arg.values()
            for arg_name, arg in zip(string.ascii_uppercase, mvs)}

    keys, exprs = tuple(res.keys()), list(res.values())
    if output_mv_idx is not None:
        keys = ()
    func = lambdify(args, exprs, funcname=funcname,
                    cse=algebra.cse, printer=printer, func_printer=func_printer,
                    output_mv_idx=output_mv_idx
                    )
    return CompiledExpression(
        algebra, keys, func, MVType or MultiVector, output_mv_idx, wrapper(func) if wrapper else func, values_asarray=values_asarray
    )

def do_compile(codegen, *tapes, wrapper=None, values_asarray=None) -> CompiledExpression:
    """ Non-symbolic compile. """
    algebra = tapes[0].algebra
    namespace = algebra.numspace

    res = codegen(*tapes)
    funcname = f'{codegen.__name__}_' + '_x_'.join(f"{tape.type_number}" for tape in tapes)
    header = f"def {funcname}({', '.join(t.expr for t in tapes)}):"
    body_lines = [f"    return {res.expr}" if not isinstance(res, str) else f"    return ({res},)"]

    func = _build_and_cache_func(header, body_lines, funcname, namespace=namespace, count_ops=False)
    return CompiledExpression(
        algebra, res.keys() if not isinstance(res, str) else (0,), func, res.mvtype, wrapped_func=wrapper(func) if wrapper else func, values_asarray=values_asarray
    )


_POW_RE = re.compile(r'\*\*\s*(\d+)?')


def _count_muls_adds(funcstr: str) -> tuple:
    """Count multiplications, divisions and additions/subtractions in a generated function string.

    :return: Tuple of (muls, divs, adds).
    """
    muls = funcstr.count('*')
    divs = funcstr.count('/')
    adds = funcstr.count('+') + funcstr.count('-')
    # Each ``**`` has been counted as two muls above, correct for that.
    for m in _POW_RE.finditer(funcstr):
        muls -= 2
        exp = m.group(1)
        muls += max(int(exp) - 1, 0) if exp is not None else 1
    return muls, divs, adds


def _op_count_str(funcstr: str) -> str:
    """Return the ``n muls / n divs / n adds`` summary for generated source. """
    muls, divs, adds = _count_muls_adds(funcstr)
    return f'{muls} muls / {divs} divs / {adds} adds'


def _compile_and_cache(func_source: str, funcname: str, namespace=None):
    """Compile complete function source, exec it, and register it with :mod:`linecache`.
    Registering the source makes it visible to :func:`inspect.getsource` and to tracebacks,
    which would otherwise have no file to read the generated code from.

    :param func_source: Complete source string of a single function.
    :param funcname: Name of the function, also used as the linecache key.
    :param namespace: Execution namespace dict. Defaults to {'builtins': builtins, 'range': range}.
    :return: The compiled function object.
    """
    if namespace is None:
        namespace = {'builtins': builtins, 'range': range}
    func_locals = {}
    exec(compile(func_source, funcname, 'exec'), namespace, func_locals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[funcname] = (len(func_source), None, func_source.splitlines(True), funcname)  # type: ignore
    return func_locals[funcname]


def _build_and_cache_func(header, body_lines, funcname, namespace=None, count_ops=True):
    """Build a function from header + body lines, insert docstring, compile, exec, cache.

    :param header: The `def funcname(...):` line.
    :param body_lines: List of indented body lines (without the docstring).
    :param funcname: Name used as the linecache key.
    :param namespace: Execution namespace dict. Defaults to {'builtins': builtins, 'range': range}.
    :param count_ops: Whether to add the `n muls / n divs / n adds` line to the docstring. Set this to
        :code:`False` when the body merely calls other generated functions, since the arithmetic then
        lives in the callees and a count of the body alone would be misleading.
    :return: The compiled function object.
    """
    count_line = []
    if count_ops:
        count_line = ['    ' + _op_count_str('\n'.join([header, *body_lines]))]
    all_lines = [header, f'    """', *count_line, f'    """'] + body_lines
    return _compile_and_cache('\n'.join(all_lines), funcname, namespace)


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
        exprs = tuple(ops.cp(a, b).values())
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

    tosympy = lambda x: x.tosympy() if hasattr(x, 'tosympy') else sympify(x)
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

    func = _compile_and_cache(funcstr, funcname, namespace)
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
        op_counts = _op_count_str('\n'.join(funclines) + '\n')
        funclines.insert(1, f'    """{op_counts}"""')

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
