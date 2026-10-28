"""
Triton kernels for an operator, from the expressions :func:`~kingdon.codegen.do_compile_symbolic` hands a lambdifier: see :doc:`backends/triton2`.

Once per compiled expression, :func:`_prepare` expands its array nodes into coefficients, takes its common subexpressions, gives every value its axes and derives its backward.
Once per variant, :class:`_Plan` cuts that into regions and Torch calls and lays out their buffers, :class:`_Printer` writes a kernel per region, and :class:`_Variant` runs them.
"""
from __future__ import annotations

import graphlib
import itertools
import math
import os
import string
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import sympy
from sympy.core.symbol import Str

from kingdon.codegen import ArrayBase, Blade, Cat, Einsum, Reduce, Reshape, Stack, Take, Unsupported, _compile_and_cache, einsum_adjoints, prepare

#: The shared memory every CUDA device grants a block: a contraction whose tile without rows needs more goes to Torch.
SHARED = 48 * 1024
#: Warps per program, twice that for a region with a tl.dot, whose 16 rows would otherwise give a thread too many elements of each tile.
#: One stage: a block's loads are all used at once, so pipelining them across blocks only holds more registers.
WARPS, STAGES = 4, 1
#: The elements of the widest coefficient tile, and of the widest broadcast product, a thread holds: more cost registers, spills and compile time.
VALUES, PRODUCTS = 1, 4
#: The longest axis unrolled into a tile per index rather than made an axis of tiles, where it would be padded to a power of two and summed across lanes.
UNROLL = 4
#: The least M, N and K of tl.dot.
DOT = 16
#: The axis of the rows, among the axes of a tile.
ROWS = -1


class sigmoid(sympy.Function):
    """ The logistic function, which torch.sigmoid of a symbolic coefficient is, as one sympy can differentiate. """
    def fdiff(self, argindex=1):
        return self * (1 - self)


#: Triton's name for every function a kernel may call, by sympy's; sign is printed apart.
FUNCTIONS = {'exp': 'tl.exp', 'log': 'tl.log', 'sin': 'tl.sin', 'cos': 'tl.cos', 'erf': 'tl.erf', 'Abs': 'tl.abs', 'sigmoid': 'tl.sigmoid', 'sign': None}
#: The functions that are zero at zero, and those that are finite wherever their argument is.
ODD = {'sin', 'erf', 'Abs', 'sign'}
BOUNDED = {'exp', 'sin', 'cos', 'erf', 'Abs', 'sigmoid', 'sign'}


@dataclass(frozen=True)
class Leaf:
    """ A coefficient of operand `arg`: its values at `index`, which fixes their blade, stacked entry and the leading axes an expansion takes entries of. """
    arg: int
    index: tuple[int, ...]


@dataclass(frozen=True)
class Shape:
    """ The axes of a value: whether it has the rows, and the classes of its named axes in order. """
    rows: bool
    named: tuple[int, ...]


@dataclass
class Prepared:
    """
    A compiled expression as a program over its leaves, and the program of its backward, each binding one array node or one pointwise expression over symbols.
    `layouts` gives, per operand, the number of leading axes of its values its coefficients are at, `prefixes` their lengths, and, for an operand with rows, the number of its named axes.
    `declared` maps each cotangent binding to the value whose shape it is summed down to, `lengths` each symbol for the length of some axes to their classes,
    `reshapes` each Reshape binding to the number of axes it keeps and, per axis after them, the axis of its operand's it is, or None for one of length one, and `dims` every value to the members of its classes.
    """
    name: str
    args: tuple[str, ...]
    prefixes: tuple[tuple[int, ...], ...]
    layouts: tuple[tuple[int, int | None], ...]
    leaves: dict[sympy.Symbol, Leaf]
    forward: list
    outputs: list
    entries: tuple[int, ...]
    cotangents: list
    backward: list
    gradients: dict
    shapes: dict[sympy.Symbol, Shape]
    declared: dict
    lengths: dict
    reshapes: dict
    dims: dict


def _nested(index):
    return tuple(map(_nested, index)) if isinstance(index, sympy.Tuple) else int(index)


def _tokens(term):
    """ The axes of an einsum term, '...' being one. """
    head, dots, tail = term.partition('...')
    return [*head, *(['...'] if dots else []), *tail]


def _objects(values):
    """ `values` as an array of objects, which numpy would otherwise take apart where it can. """
    values = list(values)
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _map(f, *args):
    """ `f` of `args`, entry by entry where any of them is an array of coefficients. """
    if not any(isinstance(a, np.ndarray) for a in args):
        return f(*args)
    return np.frompyfunc(f, len(args), 1)(*args)


class _Expansion:
    """
    The expressions of a compiled expression with every array node taken apart into coefficients, see :func:`_prepare`: each node is a coefficient, or an array of them along the axes the node puts in front.
    An einsum is one per entry of the axes its array operands put in front, of the coefficients there, over the batch dimensions only.
    """
    def __init__(self, args, shapes):
        self.leaves, self.shape, self.symbols, self.memo = {}, {}, {}, {}
        for arg, (name, values) in enumerate(args.items()):
            for blade, value in enumerate(values):
                stacked = isinstance(value, (list, tuple))
                for depth, v in (enumerate(value) if stacked else [(None, value)]):
                    v = v.tosympy() if hasattr(v, 'tosympy') else sympy.sympify(v)
                    if isinstance(v, sympy.Symbol):
                        leaf = self.symbols[v] = sympy.Symbol(v.name, real=True)
                        self.leaves[leaf] = Leaf(arg, (blade,) if depth is None else (blade, depth))
                        self.shape[leaf] = tuple(shapes[name])[1 if stacked else 0:]

    def __call__(self, e):
        if (x := self.memo.get(e)) is None:
            x = self.memo[e] = self.expand(e)
        return x

    def expand(self, e):
        if isinstance(e, sympy.Symbol):
            if e not in self.symbols:
                raise Unsupported(f'the symbol {e}, which is no coefficient of an operand')
            return self.symbols[e]
        if not e.args:
            return e
        if isinstance(e, Stack):
            parts = [self(a) for a in e.args]
            shape = np.broadcast_shapes(*(np.shape(p) for p in parts))
            array = np.empty((len(parts), *shape), dtype=object)
            for i, p in enumerate(parts):
                array[i] = p
            return array
        if isinstance(e, Cat):
            return np.concatenate([self.entries(a) for a in e.args])
        if isinstance(e, Blade):
            return self.entries(e.args[0])[int(e.args[1])]
        if isinstance(e, Take):
            return self.entries(e.args[0])[np.array(_nested(e.args[1]), dtype=int)]
        if isinstance(e, Einsum):
            return self.einsum(e.args[0].name, *e.args[1:])
        if isinstance(e, (Reduce, Reshape)):
            return _map(lambda c: e.func(c, *e.args[1:]), self(e.args[0]))
        if isinstance(e, ArrayBase):
            raise Unsupported(f'a {type(e).__name__} node')
        func = sigmoid if e.func.__name__ == 'sigmoid' else e.func
        if not isinstance(e, (sympy.Add, sympy.Mul, sympy.Pow)) and func.__name__ not in FUNCTIONS:
            raise Unsupported(f'the function {e.func.__name__}, which Triton has no kernel function for')
        if isinstance(e, sympy.Pow) and not (e.exp.is_number and (sympy.Rational(str(e.exp)) * 4).is_Integer):
            raise Unsupported(f'the power {e.exp}')
        return _map(lambda *a: func(*a), *(self(a) for a in e.args))

    def entries(self, e):
        """ `e` as an array along its first axis. """
        x = self(e)
        return x if isinstance(x, np.ndarray) else _objects(self.index(x, k) for k in range(self.prep(x)[0]))

    def index(self, c, k):
        """ Entry `k` of the coefficient `c` along its first axis, which each leaf of it either has first or broadcasts along. """
        n = len(self.prep(c))

        def at(e):
            if e in self.leaves:
                shape = self.shape[e]
                return e if len(shape) < n else self.leaf(e, k if shape[0] > 1 else 0)
            if isinstance(e, ArrayBase):
                raise Unsupported(f'an entry of a {type(e).__name__} of coefficients')
            return e.func(*map(at, e.args)) if e.args else e
        return at(c)

    def leaf(self, s, k):
        t = sympy.Symbol(f'{s.name}[{k}]', real=True)
        if t not in self.leaves:
            self.leaves[t], self.shape[t] = Leaf(self.leaves[s].arg, (*self.leaves[s].index, k)), self.shape[s][1:]
        return t

    def prep(self, e):
        """ The shape `e` has for the operands it was prepared for. """
        if e in self.shape:
            return self.shape[e]
        if not e.args or e.is_number:
            return ()
        if isinstance(e, Einsum):
            import torch
            shape = tuple(torch.einsum(e.args[0].name, *(torch.empty(self.prep(a), device='meta') for a in e.args[1:])).shape)
        elif isinstance(e, Reduce):
            x = self.prep(e.args[0])
            shape = tuple(d for i, d in enumerate(x) if i - len(x) not in map(int, e.args[2]))
        elif isinstance(e, Reshape):
            x, k, sizes = self.prep(e.args[0]), int(e.args[1]), tuple(map(int, e.args[2]))
            head, tail = x[:len(x) - k], math.prod(x[len(x) - k:])
            shape = (*head, *(tail // -math.prod(sizes) if s == -1 else s for s in sizes))
        else:
            shape = tuple(np.broadcast_shapes(*map(self.prep, e.args)))
        self.shape[e] = shape
        return shape

    def einsum(self, pattern, *operands):
        lefts, right = pattern.split('->')
        terms, out = [_tokens(t) for t in lefts.split(',')], _tokens(right)
        arrays = []
        for op in operands:
            x = self(op)
            arrays.append(x if isinstance(x, np.ndarray) else _scalar(x))
        sizes = {}
        for term, a in zip(terms, arrays):
            for letter, n in zip(term, a.shape):
                if letter == '...':
                    raise Unsupported('an array of coefficients along the batch dimensions')
                sizes[letter] = max(sizes.get(letter, 1), n)
        kept = [letter for letter in out if letter in sizes]
        if out[:len(kept)] != kept:
            raise Unsupported(f'the einsum {pattern}, whose blade axes do not lead its output')
        summed, rest = [letter for letter in sizes if letter not in out], out[len(kept):]
        result = np.empty(tuple(sizes[letter] for letter in kept), dtype=object)
        for at in np.ndindex(result.shape):
            products = []
            for over in itertools.product(*(range(sizes[letter]) for letter in summed)):
                value = dict(zip(kept, at)) | dict(zip(summed, over))
                factors = []
                for term, a in zip(terms, arrays):
                    c, t = a[tuple(value[letter] if a.shape[i] > 1 else 0 for i, letter in enumerate(term[:a.ndim]))], term[a.ndim:]
                    while t and t[0] in value:
                        c, t = self.index(c, value[t[0]]), t[1:]
                    if any(letter in value for letter in t):
                        raise Unsupported(f'the einsum {pattern}, whose blade axes are not the first of its coefficients')
                    factors.append((c, t))
                if not any(c == 0 for c, _ in factors):
                    products.append(_product(factors, rest))
            result[at] = sympy.Add(*products)
        return result[()] if not result.ndim else result


def _scalar(x):
    array = np.empty((), dtype=object)
    array[()] = x
    return array


def _product(factors, out):
    """ The einsum of the coefficients and terms `factors` into the term `out`, as a product where it contracts and transposes nothing. """
    number = sympy.Mul(*(c for c, _ in factors if c.is_number))
    factors = [(c, t) for c, t in factors if not c.is_number]
    if not factors:
        return number
    elementwise = all(letter in out for _, t in factors for letter in t) and all(t == out[len(out) - len(t):] for _, t in factors if t)
    if elementwise:
        return number * sympy.Mul(*(c for c, _ in factors))
    return number * Einsum(Str(','.join(''.join(t) for _, t in factors) + '->' + ''.join(out)), *(c for c, _ in factors))


class _Normal:
    """ Programs in which every array node, and every operand of one, is a binding of its own: each binding is one array node or one pointwise expression over symbols. """
    def __init__(self):
        self.count = itertools.count()

    def __call__(self, bindings, outputs, keep=()):
        """ The program of `bindings` and `outputs` over them; a binding in `keep` stays a symbol of its own. """
        program, memo, rename = [], {}, {}

        def bind(e):
            s = sympy.Symbol(f'_s{next(self.count)}', real=True)
            program.append((s, e))
            return s

        def atom(e):
            return e if isinstance(e, sympy.Symbol) or e.is_number else bind(e)

        def visit(e):
            if e in rename:
                return rename[e]
            if not e.args:
                return e
            if (r := memo.get(e)) is None:
                if isinstance(e, Einsum):
                    r = bind(Einsum(e.args[0], *(atom(visit(a)) for a in e.args[1:])))
                elif isinstance(e, ArrayBase):
                    r = bind(e.func(atom(visit(e.args[0])), *e.args[1:]))
                else:
                    r = e.func(*map(visit, e.args))
                memo[e] = r
            return r

        for s, e in bindings:
            rename[s] = bind(visit(e)) if s in keep else atom(visit(e))
        return program, [atom(visit(e)) for e in outputs], rename


class _Axes:
    """
    The axes of every value of a program, as classes of a union-find over the axes of all of them: the einsum letters and the broadcasting of pointwise expressions join axes,
    and those an ellipsis covers are the rows. A Reshape adds or drops axes of length one, `units`.
    """
    def __init__(self):
        self.parent, self.units, self.ellipses, self.dims, self.reshapes = [], set(), [], {}, {}

    def fresh(self):
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = i = self.parent[self.parent[i]]
        return i

    def union(self, *members):
        root = self.find(members[0])
        for m in members[1:]:
            self.parent[self.find(m)] = root

    def infer(self, program, declared=()):
        for s, e in program:
            if s in declared:
                self.dims[s] = self.dims[declared[s]]
            else:
                self.dims[s] = self.of(s, e)

    def of(self, s, e):
        if isinstance(e, Einsum):
            lefts, right = e.args[0].name.split('->')
            letters, ellipses = {}, []
            for term, op in zip(lefts.split(','), e.args[1:]):
                tokens, dims = _tokens(term), self.dims[op]
                n = len(dims) - len([t for t in tokens if t != '...'])
                at = 0
                for t in tokens:
                    if t == '...':
                        ellipses.append(dims[at:at + n])
                        at += n
                    else:
                        letters.setdefault(t, []).append(dims[at])
                        at += 1
            for members in letters.values():
                self.union(*members)
            longest = max(ellipses, key=len, default=())
            self.ellipses += [d for d in ellipses if d]
            return tuple(m for t in _tokens(right) for m in (longest if t == '...' else [letters[t][0]]))
        if isinstance(e, Reduce):
            dims = self.dims[e.args[0]]
            return tuple(d for i, d in enumerate(dims) if i - len(dims) not in map(int, e.args[2]))
        if isinstance(e, Reshape):
            dims, k, sizes = self.dims[e.args[0]], int(e.args[1]), tuple(map(int, e.args[2]))
            head, tail = dims[:len(dims) - k], dims[len(dims) - k:]
            kept = [d for d in tail if d not in self.units]
            if any(z not in (-1, 1) for z in sizes) or list(sizes).count(-1) > 1 or len(kept) > list(sizes).count(-1):
                raise Unsupported(f'a reshape into {sizes}, which only adds or drops axes of length one')
            out = []
            for z in sizes:
                out.append(kept.pop() if z == -1 and kept else self.fresh())
                if z == 1 or out[-1] not in dims:
                    self.units.add(out[-1])
            self.reshapes[s] = (len(head), tuple(None if d in self.units else tail.index(d) for d in out))
            return (*head, *out)
        dims = [self.dims[x] for x in e.free_symbols if x in self.dims]
        n = max(map(len, dims), default=0)
        out = []
        for k in range(n, 0, -1):
            members = [d[-k] for d in dims if len(d) >= k]
            self.union(*members)
            out.append(next((m for m in members if m not in self.units), members[0]))
        return tuple(out)

    def leaf(self, s, n):
        self.dims[s] = tuple(self.fresh() for _ in range(n))

    def shapes(self, flat):
        """ The Shape of every value. Without einsums, the rows of an operator are all but the last axis of its widest values, or that one axis. """
        groups = self.ellipses
        if not groups and flat:
            widest = max(map(len, flat.values()), default=0)
            groups = [d[:max(widest - 1, 1)] for d in flat.values() if widest and len(d) == widest]
        # Every value with rows has all of them, so the ellipses of all einsums are the same axes.
        longest = max(groups, key=len, default=())
        for k in range(1, len(longest) + 1):
            self.union(*(d[-k] for d in groups if len(d) >= k))
        rows = {self.find(m) for m in longest}
        shapes = {}
        for s, dims in self.dims.items():
            classes = [self.find(d) for d in dims]
            n = sum(c in rows for c in classes)
            if any(c in rows for c in classes[n:]) or n not in (0, len(rows)):
                raise Unsupported(f'a value with only some of the rows, or rows after its named axes: {s}')
            shapes[s] = Shape(bool(n), tuple(classes[n:]))
        return shapes


def _summed(c, s, x):
    """ The contribution `c` of the shape `s` to the gradient of `x`, summed over the rows and leading axes that `x` broadcasts along. """
    n, m = len(s.named), len(x.named)
    if (s.rows and not x.rows) or n > m:
        letters = string.ascii_letters[:n]
        return Einsum(Str(f"{'...' if s.rows else ''}{letters}->{'...' if x.rows else ''}{letters[n - m:]}"), c)
    return c


def _einsum_adjoints(e, d):
    """ The gradient of each operand of the Einsum `e` by its cotangent `d`, broadcast over what the transposed einsum cannot give it: the rows, and trailing axes. """
    for op, adjoint in einsum_adjoints(e, d):
        lefts, right = adjoint.args[0].name.split('->')
        have = set(lefts.replace('...', '').replace(',', ''))
        out = [t for t in _tokens(right) if t != '...' or '...' in lefts]
        missing = [t for t in out if t != '...' and t not in have]
        if out[len(out) - len(missing):] != missing:
            raise Unsupported(f'the gradient of the einsum {e.args[0].name}, which broadcasts along axes before others')
        kept = out[:len(out) - len(missing)]
        adjoint = Einsum(Str(f"{lefts}->{''.join(kept)}"), *adjoint.args[1:])
        if missing:
            adjoint = Reshape(adjoint, 1, sympy.Tuple(-1, *[1] * len(missing))) if kept else Reshape(adjoint, 0, sympy.Tuple(*[1] * len(missing)))
        yield op, adjoint


def _prepare(args, exprs, funcname, cse, shapes, codegen_symbolcls):
    """ The :class:`Prepared` of a compiled expression, of operands `args` of `shapes` and output coefficients `exprs`. """
    tosympy = lambda x: x.tosympy() if hasattr(x, 'tosympy') else sympy.sympify(x)
    expansion = _Expansion(args, shapes)
    if any(isinstance(e, sympy.Basic) and e.has(ArrayBase) for e in exprs):
        expanded = [expansion(e) for e in exprs]
        entries = {np.shape(x) for x in expanded}
        if len(entries) > 1:
            raise Unsupported('outputs of different shapes of coefficients')
        entries = entries.pop()
        outputs = [c for x in expanded for c in (x.flat if isinstance(x, np.ndarray) else [x])]
        bindings, outputs = prepare(outputs, sympy.Symbol) if cse else ([], outputs)
    else:
        entries = ()
        bindings, outputs = prepare(list(exprs), codegen_symbolcls) if cse else ([], list(exprs))
        expanded = []
        for name, e in bindings:
            name = sympy.Symbol(str(name))
            expansion.symbols[name] = name
            expanded.append((name, expansion(tosympy(e))))
        bindings, outputs = expanded, [expansion(tosympy(e)) for e in outputs]
    normal = _Normal()
    forward, outputs, _ = normal(bindings, outputs)
    used = {x for _, e in forward for x in e.free_symbols} | set(outputs)
    leaves = {leaf: expansion.leaves[leaf] for leaf in expansion.leaves if leaf in used}

    axes = _Axes()
    for leaf in leaves:
        axes.leaf(leaf, len(expansion.shape[leaf]))
    axes.infer(forward)
    flat = None if any(isinstance(e, ArrayBase) for _, e in forward) else {leaf: axes.dims[leaf] for leaf in leaves}
    shapes_ = axes.shapes(flat)

    # A mean is a sum times one over the length of its axes, and a sum an einsum, whose gradient is its transpose.
    lengths, rewritten = {}, []
    for s, e in forward:
        if isinstance(e, Reduce):
            x, operation, at = e.args
            if operation.name not in ('sum', 'mean'):
                raise Unsupported(f'the reduction {operation.name}, whose gradient Triton has no kernel for')
            xs, dims = shapes_[x], axes.dims[x]
            offset = len(dims) - len(xs.named)
            reduced = {len(dims) + int(a) for a in at}
            rows = any(i < offset for i in reduced)
            if rows and not all(i in reduced for i in range(offset)):
                raise Unsupported('a reduction over some of the rows')
            letters = string.ascii_letters[:len(xs.named)]
            kept = ''.join(letter for i, letter in enumerate(letters) if offset + i not in reduced)
            e = Einsum(Str(f"{'...' if xs.rows else ''}{letters}->{'...' if xs.rows and not rows else ''}{kept}"), x)
            if operation.name == 'mean':
                total = sympy.Symbol(f'_n{len(lengths)}', positive=True)
                lengths[total] = (*((ROWS,) if rows else ()), *(c for i, c in enumerate(xs.named) if offset + i in reduced))
                summed = sympy.Symbol(f'{s.name}_sum', real=True)
                axes.dims[summed] = axes.dims[s]
                rewritten += [(summed, e), (s, summed / total)]
                continue
        rewritten.append((s, e))
    forward = rewritten
    for total in lengths:
        axes.leaf(total, 0)

    # The cotangent of an output has the shape of all of them, which its own broadcasts to.
    rowed = next((o for o in outputs if isinstance(o, sympy.Symbol) and shapes_[o].rows), None)
    widest = max((o for o in outputs if isinstance(o, sympy.Symbol)), key=lambda o: len(shapes_[o].named), default=None)
    cotangents = [sympy.Symbol(f'_g{k}', real=True) for k in range(len(outputs))]
    for g, o in zip(cotangents, outputs):
        if not isinstance(o, sympy.Symbol):
            axes.leaf(g, 0)
            continue
        named = axes.dims[o][len(axes.dims[o]) - len(shapes_[o].named):]
        wide = axes.dims[widest][len(axes.dims[widest]) - len(shapes_[widest].named):]
        rows = axes.dims[rowed][:len(axes.dims[rowed]) - len(shapes_[rowed].named)] if rowed is not None else ()
        axes.dims[g] = (*rows, *wide[:len(wide) - len(named)], *named)
    shapes_ = axes.shapes(flat)
    adjoints, backward, declared = defaultdict(list), [], {}
    for g, o in zip(cotangents, outputs):
        if isinstance(o, sympy.Symbol):
            adjoints[o].append(_summed(g, shapes_[g], shapes_[o]))
    for s, e in reversed(forward):
        if s not in adjoints:
            continue
        d = sympy.Symbol(f'_d{s.name}', real=True)
        backward.append((d, sympy.Add(*adjoints.pop(s))))
        declared[d] = s
        if isinstance(e, Einsum):
            contributions = _einsum_adjoints(e, d)
        elif isinstance(e, Reshape):
            x = e.args[0]
            head, out = axes.reshapes[s]
            contributions = [(x, Reshape(d, len(out), sympy.Tuple(*(1 if m in axes.units else -1 for m in axes.dims[x][head:]))))]
        else:
            contributions = []
            for x in e.free_symbols:
                if x in lengths or x not in shapes_:
                    continue
                derivative = sympy.diff(e, x)
                if derivative != 0:
                    contributions.append((x, _summed(d * derivative.xreplace({e: s}), shapes_[s], shapes_[x])))
        for x, c in contributions:
            if isinstance(x, sympy.Symbol):
                adjoints[x].append(c)
    gradients = {}
    for leaf in leaves:
        g = sympy.Symbol(f'_d{leaf.name}', real=True)
        backward.append((g, sympy.Add(*adjoints.get(leaf, []))))
        gradients[leaf], declared[g] = g, leaf
    # The provider shares what the cotangents have in common: a binding of it comes after the cotangents it is over, and before those over it.
    cses, reduced = prepare([e for _, e in backward], sympy.Symbol) if cse else ([], [e for _, e in backward])
    order = dict(cses) | {d: e for (d, _), e in zip(backward, reduced)}
    graph = {s: {x for x in e.free_symbols if x in order} for s, e in order.items()}
    backward, _, rename = normal([(s, order[s]) for s in graphlib.TopologicalSorter(graph).static_order()], [], keep=declared)
    declared = {rename[d]: s for d, s in declared.items()}
    gradients = {leaf: rename[g] for leaf, g in gradients.items()}
    axes.infer(backward, declared)
    shapes_ = axes.shapes(flat)
    layouts, prefixes = [], []
    for arg, name in enumerate(args):
        own = [leaf for leaf in leaves if leaves[leaf].arg == arg]
        lead = max((len(leaves[leaf].index) for leaf in own), default=0)
        rows = any(shapes_[leaf].rows for leaf in own)
        trail = {len(shapes_[leaf].named) for leaf in own if shapes_[leaf].rows}
        if rows and (len(trail) > 1 or any(len(leaves[leaf].index) != lead for leaf in own)):
            raise Unsupported('coefficients of one operand with different axes')
        if any(leaves[a].index != leaves[b].index and leaves[b].index[:len(leaves[a].index)] == leaves[a].index for a in own for b in own):
            raise Unsupported('a coefficient of an operand used both whole and entry by entry')
        layouts.append((lead, trail.pop() if rows else None))
        prefixes.append((len(args[name]), *tuple(shapes[name])[:lead - 1]) if lead else ())
    return Prepared(funcname, tuple(args), tuple(prefixes), tuple(layouts), leaves, forward, outputs, entries,
                    cotangents, backward, gradients, shapes_, declared, lengths, axes.reshapes, axes.dims)


@dataclass(frozen=True)
class Access:
    """
    Where a coefficient is in the tensor `buffer`, in elements: at `offset`, which is c0 + c1 R, plus `rows` times its row unless it has none,
    plus per (position, class, stride) of its named axes longer than one the stride times its index. It has `named` named axes in all.
    """
    buffer: str
    offset: tuple[int, int]
    rows: int | None
    axes: tuple[tuple[int, int, int], ...]
    named: int


@dataclass(frozen=True)
class Buffer:
    """ A tensor a step makes, of `shape` with the rows inserted at axis `rows` unless it has none. A `partial` one has an entry per program in front, which the step sums. Under a false `flag` it is left empty. """
    name: str
    shape: tuple[int, ...]
    rows: int | None
    partial: bool = False
    flag: str | None = None


@dataclass(frozen=True)
class Accumulator:
    """ A sum over the rows of a region into `access`, of each (coefficient, einsum binding) of `terms` block by block, and of `rest` once. """
    terms: tuple
    rest: sympy.Expr
    access: Access
    flag: str | None


@dataclass
class Region:
    """ One kernel: block of rows by block of rows, it loads `loads`, computes `bindings`, stores `stores` and accumulates `accumulators`, reading the buffers `reads` and making `writes`. """
    name: str
    rows: bool
    loads: dict
    bindings: list
    stores: list
    accumulators: list
    reads: list
    writes: list
    flags: tuple


@dataclass(frozen=True)
class TorchCall:
    """ torch.einsum of `pattern` over `operands`, each (buffer, index) of a view of one, into `result`. """
    pattern: str
    operands: tuple
    result: Buffer
    flag: str | None


def _operands(e):
    """ The symbols a binding is over. """
    if isinstance(e, Einsum):
        return [a for a in e.args[1:] if isinstance(a, sympy.Symbol)]
    if isinstance(e, Reshape):
        return [e.args[0]]
    return [x for x in e.free_symbols if isinstance(x, sympy.Symbol)]


def _letters(e):
    """ The letters of the named axes of each operand of the Einsum `e`, and of its output. """
    lefts, right = e.args[0].name.split('->')
    return [[t for t in _tokens(term) if t != '...'] for term in lefts.split(',')], [t for t in _tokens(right) if t != '...']


def _aligned(full, n):
    """ The last `n` of `full`, right aligned, padded with False in front. """
    return ((False,) * n + tuple(full))[len(full):] if n else ()


def _pow2(n):
    return 1 << max(n - 1, 0).bit_length()


def _contiguous(shape):
    """ The strides of a contiguous tensor of `shape`, each c0 + c1 R as (c0, c1) when the shape holds an R, as None. """
    strides, stride = [], (1, 0)
    for n in reversed(shape):
        strides.append(stride)
        stride = (0, stride[0] + stride[1]) if n is None else (stride[0] * n, stride[1] * n)
    return strides[::-1], stride


class _Plan:
    """
    A compiled expression cut into regions and Torch calls for operands of one variant, with the layouts of their buffers: every coefficient is an :class:`Access` of one.
    The backward is planned first, so that the forward saves exactly the values its backward loads.
    """
    def __init__(self, p: Prepared, values, itemsize, budgets=(SHARED, SHARED)):
        self.p, self.itemsize, self.names = p, itemsize, itertools.count()
        self.length, self.full, ell = {}, {}, None
        for leaf, l in p.leaves.items():
            shape, coefficient = p.shapes[leaf], tuple(values[l.arg].shape[len(l.index):])
            n = len(shape.named)
            named = coefficient[len(coefficient) - n:] if n else ()
            if shape.rows:
                if ell is not None and coefficient[:len(coefficient) - n] != ell:
                    raise Unsupported(f'operands of different rows, {ell} and {coefficient[:len(coefficient) - n]}')
                ell = coefficient[:len(coefficient) - n]
            elif len(coefficient) != n:
                raise Unsupported(f'an operand of {len(coefficient)} batch dimensions where it was prepared for {n}')
            for c, z in zip(shape.named, named):
                if z != 1 and self.length.get(c, 1) not in (1, z):
                    raise Unsupported(f'axes of lengths {self.length[c]} and {z} that do not broadcast')
                self.length[c] = max(self.length.get(c, 1), z)
            self.full[leaf] = (shape.rows, tuple(z > 1 for z in named))
        self.values = [tuple(v.shape) for v in values]
        for arg, (prefix, (lead, _)) in enumerate(zip(p.prefixes, p.layouts)):
            if self.values[arg][:lead] != prefix:
                raise Unsupported(f'operand {p.args[arg]} of {self.values[arg][:lead]} blades and entries, where {p.name} was prepared for {prefix}')
        for c in {c for s in p.shapes.values() for c in s.named}:
            self.length.setdefault(c, 1)
        self.block = {c: _pow2(n) for c, n in self.length.items()}
        for total in p.lengths:
            self.full[total] = (False, ())
        for s, e in p.forward:
            self.full[s] = self.effect(s, e)
        self.outputs = [o for o in p.outputs if isinstance(o, sympy.Symbol)]
        named = max((len(p.shapes[o].named) for o in self.outputs), default=0)
        self.out_rows = any(self.full[o][0] for o in self.outputs)
        widest = max(self.outputs, key=lambda o: len(p.shapes[o].named), default=None)
        self.out_classes = p.shapes[p.cotangents[p.outputs.index(widest)]].named if widest is not None else ()
        self.out_lengths = [1] * named
        for o in self.outputs:
            for q, (c, f) in enumerate(zip(p.shapes[o].named, self.full[o][1])):
                k = named - len(p.shapes[o].named) + q
                self.out_lengths[k] = max(self.out_lengths[k], self.length[c] if f else 1)
        for g, o in zip(p.cotangents, p.outputs):
            if isinstance(o, sympy.Symbol):
                n = len(p.shapes[g].named)
                self.full[g] = (self.out_rows and p.shapes[g].rows, tuple(z > 1 for z in self.out_lengths[named - n:]))
            else:
                self.full[g] = (False, ())
        for s, e in p.backward:
            self.full[s] = self.effect(s, e)
        self.kind, self.together = {}, {}
        for program, budget in zip((p.forward, p.backward), budgets):
            for s, e in program:
                self.kind[s] = self.classify(s, e)
            self.budget(program, budget)

    def effect(self, s, e):
        """ Whether `s` has the rows, and which of its named axes are longer than one, from those of what it is over. """
        p, shape = self.p, self.p.shapes[s]
        if isinstance(e, Einsum):
            terms, out = _letters(e)
            longer = {t for o, term in zip(e.args[1:], terms) for t, f in zip(term, self.full[o][1]) if f}
            return shape.rows and any(self.full[o][0] for o in e.args[1:]), tuple(t in longer for t in out)
        if isinstance(e, Reshape):
            x = e.args[0]
            head, out = p.reshapes[s]
            rows_x, rows_s = len(p.dims[x]) - len(p.shapes[x].named), len(p.dims[s]) - len(shape.named)
            full = []
            for q in range(len(shape.named)):
                i = rows_s + q
                j = i if i < head else None if out[i - head] is None else head + out[i - head]
                full.append(j is not None and j >= rows_x and self.full[x][1][j - rows_x])
            return self.full[x][0], tuple(full)
        rows, full = self.broadcast(e)
        if s in p.declared:
            target_rows, target = self.full[p.declared[s]]
            return rows and target_rows, tuple(a and b for a, b in zip(_aligned(full, len(target)), target))
        return rows, _aligned(full, len(shape.named))

    def broadcast(self, e):
        """ Whether the pointwise `e` has the rows, and which of its axes, right aligned, are longer than one. """
        ops = [x for x in _operands(e) if x in self.full]
        n = max((len(self.full[x][1]) for x in ops), default=0)
        full = [False] * n
        for x in ops:
            f = self.full[x][1]
            for q, v in enumerate(f):
                full[n - len(f) + q] |= v
        return any(self.full[x][0] for x in ops), tuple(full)

    def dims(self, s):
        """ The classes of the axes of the tile of `s`: the rows, then its named axes longer than one. """
        return tuple(c for _, c in self.axes(s))

    def axes(self, s):
        """ The axes of the tile of `s`, each (its position among the named axes of `s`, or 'r' for the rows; its class). """
        rows, full = self.full[s]
        return [('r', ROWS)] * rows + [(q, c) for q, (c, f) in enumerate(zip(self.p.shapes[s].named, full)) if f]

    def access(self, s, name, offset):
        """ Where `s` is in a contiguous buffer `name` of its tiles, at `offset`. """
        lengths = self.tile_lengths(s)
        strides, _ = _contiguous(lengths)
        axes = tuple((q, c, st[0]) for (q, c), st in zip([a for a in self.axes(s) if a[0] != 'r'], strides))
        return Access(name, offset, math.prod(lengths) if self.full[s][0] else None, axes, len(self.p.shapes[s].named))

    def numel(self, s):
        return math.prod(self.block[d] for d in self.dims(s) if d != ROWS)

    def classify(self, s, e):
        if isinstance(e, Reshape):
            return 'alias'
        if not isinstance(e, Einsum):
            return 'point'
        ops = [a for a in e.args[1:] if isinstance(a, sympy.Symbol)]
        if any(self.full[o][0] for o in ops) and not self.full[s][0]:
            return 'reduce' if self.numel(s) * self.itemsize <= SHARED else 'torch'
        rowless = [o for o in ops if not self.full[o][0]]
        size = max((self.numel(o) for o in rowless), default=0) if self.full[s][0] else max(self.numel(o) for o in [*ops, s])
        return 'contract' if size * self.itemsize <= SHARED else 'torch'

    def budget(self, program, budget):
        """
        A kernel keeps the operands without rows of its contractions, their weights, in shared memory for all its rows. While those of a pass come to more than `budget`,
        the contractions of one weight after another, in order, go to Torch. Every contraction Torch does is one einsum with the others of its group:
        those of one pattern and the same operands without rows, such as the blades of one grade through one weight, stacked along an axis of their own.
        """
        groups = defaultdict(list)
        for s, e in program:
            if isinstance(e, Einsum) and self.kind[s] in ('contract', 'torch'):
                groups[e.args[0].name, tuple(o for o in e.args[1:] if not self.full[o][0]), self.full[s][0]].append(s)
        weights = lambda keys: {o for key in keys if key[2] and self.kind[groups[key][0]] == 'contract' for o in key[1]}
        kept = [key for key in groups if key[1]]
        while kept and sum(self.numel(o) for o in weights(kept)) * self.itemsize > budget:
            for s in groups[kept.pop(0)]:
                self.kind[s] = 'torch'
        for key, members in groups.items():
            for s in members:
                if self.kind[s] == 'torch':
                    self.together[s] = [m for m in members if self.kind[m] == 'torch']

    def stages(self, program, uses):
        """
        The stage at which each value of `program` exists: a row reduction or a Torch call exists one stage after what it is over.
        A sum over the rows of reductions, times coefficients without rows, is accumulated as a whole: `accumulate` maps it to its terms and the rest.
        """
        avail, accumulate, folded = {}, {}, Counter()
        for s, e in program:
            avail[s] = max((avail.get(o, 0) for o in _operands(e)), default=0) + (self.kind[s] in ('reduce', 'torch'))
            if self.kind[s] == 'reduce':
                accumulate[s] = (((sympy.S.One, e, s),), sympy.S.Zero)
            if self.kind[s] != 'point' or self.full[s][0]:
                continue
            reduces = [x for x in _operands(e) if self.kind.get(x) in ('reduce', 'accum') and x in accumulate]
            if not reduces:
                continue
            rest, dependent = e.as_independent(*reduces, as_Add=True)
            terms = [t.as_independent(*reduces, as_Add=False) for t in sympy.Add.make_args(dependent)]
            k = avail[s] - 1
            if not all(r in reduces and avail[r] - 1 == k for _, r in terms):
                continue
            expanded = []
            for c, r in terms:
                inner, inner_rest = accumulate[r]
                expanded += [(c * d, einsum, reduction) for d, einsum, reduction in inner]
                rest += c * inner_rest
            others = rest.free_symbols.union(*(c.free_symbols for c, _, _ in expanded))
            if all(avail.get(x, 0) <= k for x in others):
                accumulate[s] = (tuple(expanded), rest)
                self.kind[s] = 'accum'
                folded.update(r for _, r in terms)
        # A reduction every use of which accumulates it needs no accumulator of its own.
        for r, n in folded.items():
            if n == uses[r]:
                self.kind[r] = 'folded'
                del accumulate[r]
        return avail, accumulate

    def plan(self):
        """ The steps of the forward and of the backward, and the forward buffers the backward reads. """
        p = self.p
        forward, backward = dict(p.forward), dict(p.backward)
        uses = Counter(o for program in (p.forward, p.backward) for _, e in program for o in _operands(e)) + Counter(o for o in p.outputs if isinstance(o, sympy.Symbol))
        uses += Counter(g for g in p.gradients.values() if isinstance(g, sympy.Symbol))
        self.avail, self.accumulate = {}, {}
        for program in (p.forward, p.backward):
            avail, accumulate = self.stages(program, uses)
            self.avail |= avail
            self.accumulate |= accumulate
        self.where, self.buffers, self.saves, self.slot, self.stacked = {}, {}, {}, {}, {}
        for leaf, l in p.leaves.items():
            self.where[leaf] = self.leaf_access(leaf, f'a{l.arg}')
        self.cotangent = {g: k for k, g in enumerate(p.cotangents)}
        for k, g in enumerate(p.cotangents):
            self.where[g] = self.out_access(k, 'gout')
        backward_steps = self.schedule(backward, True)
        forward_steps = self.schedule(forward, False)
        return self.records(*forward_steps, False), self.records(*backward_steps, True), list(dict.fromkeys(self.where[s].buffer for s in self.saves))

    def schedule(self, binding, backward):
        """ The regions and Torch calls of one pass, from its last stage down, so that each region knows what later ones load from it. """
        p, kind, avail = self.p, self.kind, self.avail
        sinks, calls, needed = defaultdict(list), defaultdict(list), set()
        forward = dict(p.forward)

        def accumulated(s):
            return self.accumulate.get(s, ((), s))

        def need(s, flag=None):
            if (s, flag) in needed:
                return
            needed.add((s, flag))
            if kind[s] in ('reduce', 'accum'):
                sinks[avail[s] - 1].append(('acc', s, ('q', s), flag))
            elif kind[s] == 'torch':
                members = [m for m in self.together[s] if avail[m] == avail[s]]
                if any((m, flag) in needed for m in members if m != s):
                    return
                needed.update((m, flag) for m in members)
                calls[avail[s]].append((members, flag))
                for k, column in enumerate(zip(*(binding[m].args[1:] for m in members))):
                    stacked = self.full[column[0]][0]
                    if stacked and self.stackable(column) is not None:
                        continue
                    for j, o in enumerate(column if stacked else column[:1]):
                        target = ('g', members[0], k, j) if stacked else ('m', o)
                        if o in p.leaves or o in self.cotangent:
                            if stacked:
                                sinks[0].append(('keep', o, target, flag))
                        elif backward and o in forward:
                            if stacked or self.kind[o] not in ('contract', 'torch', 'reduce', 'accum'):
                                sinks[0].append(('keep', o, target, flag))
                            else:
                                self.saves[o] = None
                        elif kind[o] in ('reduce', 'accum', 'torch'):
                            need(o, flag)
                            if stacked:
                                sinks[avail[o]].append(('keep', o, target, flag))
                        else:
                            sinks[avail[o]].append(('keep', o, target, flag))
            else:
                sinks[avail[s]].append(('keep', s, ('m', s), flag))

        def rowless(owners, values, targets, flag):
            """ `values` without rows into one partial buffer: accumulated where they are summed if that is one stage, else all put in at the last, once every one exists. """
            stages = {avail.get(v, 0) - (isinstance(v, sympy.Symbol) and v in self.accumulate) for v in values}
            if len(stages) == 1:
                sinks[stages.pop()] += [('acc', v, t, flag) for v, t in zip(values, targets)]
            else:
                sinks[max(avail.get(v, 0) for v in values)] += [('put', v, t, flag) for v, t in zip(values, targets)]

        if backward:
            for arg in range(len(p.args)):
                own = [leaf for leaf in p.leaves if p.leaves[leaf].arg == arg]
                if not own:
                    continue
                flag = f'G{arg}'
                if p.layouts[arg][1] is not None:
                    stage = max(avail.get(p.gradients[leaf], 0) for leaf in own)
                    sinks[stage] += [('store', p.gradients[leaf], ('d', leaf), flag) for leaf in own]
                    sinks[stage] += [('zero', None, ('dz', arg, index), flag) for index in self.uncovered(arg, own)]
                else:
                    rowless(own, [p.gradients[leaf] for leaf in own], [('d', leaf) for leaf in own], flag)
        else:
            for s in list(self.saves):
                need(s, 'SAVE')
            if self.out_rows:
                stage = max((avail.get(o, 0) for o in self.outputs), default=0)
                sinks[stage] += [('store', o, ('out', k), None) for k, o in enumerate(p.outputs)]
            else:
                rowless(p.outputs, p.outputs, [('out', k) for k in range(len(p.outputs))], None)

        regions = {}
        for stage in range(max(sinks, default=0), -1, -1):
            if not sinks[stage]:
                continue
            order, loads, seen = [], [], set()

            def visit(s):
                if not isinstance(s, sympy.Symbol) or s in seen or s in p.lengths:
                    return
                seen.add(s)
                if s in p.leaves or s in self.where and s not in binding and s not in forward:
                    loads.append(s)
                    return
                if backward and s in forward:
                    if self.kind[s] in ('contract', 'torch', 'reduce', 'accum'):
                        self.saves[s] = None
                        loads.append(s)
                        return
                    e = forward[s]
                else:
                    e = binding[s]
                    if kind[s] in ('torch', 'reduce', 'accum') or kind[s] == 'contract' and avail[s] < stage:
                        loads.append(s)
                        need(s)
                        return
                for o in _operands(e):
                    visit(o)
                order.append((s, e))

            for what, s, _, _ in sinks[stage]:
                if what == 'acc':
                    terms, rest = accumulated(s)
                    for c, e, _ in terms:
                        for x in [*c.free_symbols, *_operands(e)]:
                            visit(x)
                    for x in getattr(rest, 'free_symbols', ()):
                        visit(x)
                else:
                    visit(s)
            regions[stage] = (list(sinks[stage]), order, loads)
        return regions, calls

    def leaf_access(self, leaf, buffer, index=None):
        """ Where `leaf`, or the coefficient at `index` of the same operand, is in that operand's values, or in their gradient `buffer`, which has their layout. """
        l, shape = self.p.leaves[leaf], self.p.shapes[leaf]
        index = l.index if index is None else index
        values, n = self.values[l.arg], len(shape.named)
        dims = [*values[:len(index)], *([None] if shape.rows else []), *(values[len(values) - n:] if n else ())]
        strides, _ = _contiguous(dims)
        offset = (sum(i * s[0] for i, s in zip(index, strides)), sum(i * s[1] for i, s in zip(index, strides)))
        named = strides[len(dims) - n:] if n else []
        lengths = values[len(values) - n:] if n else ()
        return Access(buffer, offset, strides[len(index)][0] if shape.rows else None, tuple((q, c, st[0]) for q, (c, z, st) in enumerate(zip(shape.named, lengths, named)) if z > 1), n)

    def out_access(self, k, buffer):
        """ Where output coefficient `k` is in the output, or its cotangent in `buffer`, which has the layout of the output. """
        classes = self.p.shapes[self.p.cotangents[k]].named if isinstance(self.p.outputs[k], sympy.Symbol) else self.out_classes
        strides, _ = _contiguous([len(self.p.outputs), *([None] if self.out_rows else []), *self.out_lengths])
        named = strides[len(strides) - len(self.out_lengths):] if self.out_lengths else []
        return Access(buffer, (k * strides[0][0], k * strides[0][1]), strides[1][0] if self.out_rows else None,
                      tuple((q, c, st[0]) for q, (c, z, st) in enumerate(zip(classes, self.out_lengths, named)) if z > 1), len(self.out_lengths))

    def uncovered(self, arg, own):
        """ The indices of the coefficients of operand `arg` that none of its leaves `own` is: their gradients are zero. """
        covered = {self.p.leaves[leaf].index for leaf in own}
        return [index for index in itertools.product(*map(range, self.values[arg][:self.p.layouts[arg][0]])) if index not in covered]

    def tile_lengths(self, s):
        return tuple(self.length[d] for d in self.dims(s) if d != ROWS)

    def records(self, regions, calls, backward):
        """ The steps of one pass, in order, once every buffer of it has its layout. """
        p, tag = self.p, 'b' if backward else 'f'
        groups, own, stacks = defaultdict(list), dict(p.backward if backward else p.forward), defaultdict(dict)
        # A value the forward needs as well as its backward is made whether or not the backward will run.
        flags = defaultdict(set)
        for sinks, _, _ in regions.values():
            for what, s, target, flag in sinks:
                flags[what, s, target].add(flag)
        for stage in regions:
            sinks = list(dict.fromkeys((what, s, target, None if None in flags[what, s, target] else flag) for what, s, target, flag in regions[stage][0]))
            regions[stage] = (sinks, *regions[stage][1:])
        for stage, (sinks, _, _) in regions.items():
            for what, s, target, flag in sinks:
                if what == 'keep' and target[0] == 'g':
                    stacks[target[1], target[2]][target[3]] = (s, flag)
                elif what == 'keep' and (s not in self.where or backward and s not in own):
                    groups[('m', stage, flag, self.full[s][0], self.tile_lengths(s))].append(s)
                elif what == 'acc' and target[0] == 'q' and s not in self.where:
                    groups[('q', stage, flag, False, self.tile_lengths(s))].append(s)
        buffers = {}
        for (kind, stage, flag, rows, lengths), members in groups.items():
            name = f'{kind}{tag}{next(self.names)}'
            numel = math.prod(lengths)
            buffers[name] = Buffer(name, (len(members), *lengths), 1 if rows else None, kind == 'q', flag)
            for slot, s in enumerate(members):
                self.where[s] = self.access(s, name, (0, slot * numel) if rows else (slot * numel, 0))
                self.slot[s] = (slot,)
        for (head, k), column in stacks.items():
            first, flag = column[0]
            name = f'g{tag}{next(self.names)}'
            lengths = self.tile_lengths(first)
            numel = math.prod(lengths)
            buffers[name] = Buffer(name, (len(column), *lengths), 1 if self.full[first][0] else None, False, flag)
            for j, (o, _) in column.items():
                self.stacked[head, k, j] = self.access(o, name, (0, j * numel) if self.full[first][0] else (j * numel, 0))
            self.stacked[head, k] = name
        steps = []
        for stage in sorted(set(regions) | set(calls)):
            for members, flag in calls.get(stage, []):
                steps.append(self.call(members, flag, backward, buffers, tag))
            if stage not in regions:
                continue
            sinks, order, loads = regions[stage]
            stores, accumulators = [], []
            for what, s, target, flag in sinks:
                if what in ('acc', 'put'):
                    terms, rest = self.accumulate.get(s, ((), s)) if what == 'acc' else ((), s)
                    accumulators.append(Accumulator(terms, rest, self.target(target), flag))
                elif what == 'zero':
                    own = next(leaf for leaf in p.leaves if p.leaves[leaf].arg == target[1])
                    stores.append((sympy.S.Zero, self.leaf_access(own, f'd{target[1]}', target[2]), flag))
                elif target[0] == 'g':
                    stores.append((s, self.stacked[target[1:]], flag))
                else:
                    stores.append((s, self.where[s] if what == 'keep' else self.target(target), flag))
            accesses = [a for _, a, _ in stores] + [a.access for a in accumulators]
            writes = list(dict.fromkeys(a.buffer for a in accesses))
            for name in writes:
                if name not in buffers:
                    buffers[name] = self.fixed(name)
            reads = list(dict.fromkeys(self.where[s].buffer for s in loads if self.where[s].buffer not in writes)) or ['a0']
            rows = any(self.full[s][0] for s in loads) or any(a.rows is not None for a in accesses) or any(a.terms for a in accumulators)
            flags = tuple(sorted({f for _, _, f in stores if f} | {a.flag for a in accumulators if a.flag}))
            steps.append(Region(f'{p.name}_{tag}{stage}_{next(_OPS)}', rows, {s: self.where[s] for s in loads}, order, stores, accumulators, reads, [buffers[n] for n in writes], flags))
        self.buffers.update(buffers)
        return steps

    def stackable(self, column):
        """ Where `column`, operands of the contractions of one group, already lie one after another: the coefficients of consecutive blades of an operand, or the cotangents of consecutive outputs. """
        p = self.p
        if all(o in self.cotangent for o in column):
            first = self.cotangent[column[0]]
            if [self.cotangent[o] for o in column] == list(range(first, first + len(column))):
                return 'gout', (slice(first, first + len(column)),)
        if all(o in p.leaves for o in column):
            index = [p.leaves[o].index for o in column]
            first = index[0]
            if all(i[1:] == first[1:] and i[0] == first[0] + j and p.leaves[o].arg == p.leaves[column[0]].arg for j, (i, o) in enumerate(zip(index, column))):
                return f'a{p.leaves[column[0]].arg}', (slice(first[0], first[0] + len(column)), *first[1:])
        return None

    def call(self, members, flag, backward, buffers, tag):
        """ The one einsum of a group of contractions, its operands with rows stacked along a new first axis, its operands without rows shared. """
        p = self.p
        binding = dict(p.backward if backward else p.forward)
        e, n = binding[members[0]], len(members)
        letters, out = _letters(e)
        new = next(c for c in string.ascii_letters if c not in e.args[0].name)
        operands, terms = [], []
        for k, (term, column) in enumerate(zip(letters, zip(*(binding[m].args[1:] for m in members)))):
            o = column[0]
            stacked = self.full[o][0]
            view = self.stackable(column) if stacked else None
            if o in self.cotangent and (view is not None or not stacked):
                buffer, index = view or ('gout', (self.cotangent[o],))
                shape, rows, letters_ = tuple(self.out_lengths), self.out_rows, term
            elif o in p.leaves and (view is not None or not stacked):
                l = p.leaves[o]
                buffer, index = view or (f'a{l.arg}', l.index)
                shape = tuple(self.values[l.arg][len(self.values[l.arg]) - len(p.shapes[o].named):]) if p.shapes[o].named else ()
                rows, letters_ = p.shapes[o].rows, term
            else:
                if stacked:
                    buffer, index = self.stacked[members[0], k], ()
                else:
                    buffer, index = self.where[o].buffer, self.slot[o]
                shape, rows = self.tile_lengths(o), self.full[o][0]
                letters_ = [t for t, f in zip(term, self.full[o][1]) if f]
            operands.append((buffer, index, ((n,) if stacked else ()) + ((None,) if rows else ()) + tuple(shape)))
            terms.append((new if stacked else '') + ('...' if rows else '') + ''.join(letters_))
        s = members[0]
        lengths = self.tile_lengths(s)
        name = f't{tag}{next(self.names)}'
        result = buffers[name] = Buffer(name, (n, *lengths), 1 if self.full[s][0] else None, False, flag)
        numel = math.prod(lengths)
        for j, m in enumerate(members):
            self.where[m], self.slot[m] = self.access(m, name, (0, j * numel) if self.full[s][0] else (j * numel, 0)), (j,)
        output = new + ('...' if self.full[s][0] else '') + ''.join(t for t, f in zip(out, self.full[s][1]) if f)
        return TorchCall(f"{','.join(terms)}->{output}", tuple(operands), result, flag)

    def target(self, target):
        if target[0] == 'out':
            return self.out_access(target[1], 'out')
        if target[0] == 'd':
            return self.leaf_access(target[1], f'd{self.p.leaves[target[1]].arg}')
        return self.where[target[1]]

    def fixed(self, name):
        """ The output, or the gradient of an operand: tensors of the shape the caller sees, up to the rows. """
        if name == 'out':
            return Buffer('out', (len(self.p.outputs), *self.out_lengths), 1 if self.out_rows else None, not self.out_rows)
        arg = int(name[1:])
        lead, named = self.p.layouts[arg]
        values = self.values[arg]
        if named is None:
            return Buffer(name, values, None, True, f'G{arg}')
        return Buffer(name, (*values[:lead], *(values[len(values) - named:] if named else ())), lead, False, f'G{arg}')


def _fit(code, have, want):
    """ `code`, a tile of axes `have`, as one of the axes `want`: in their order, with a unit axis for each it lacks. """
    have = list(have)
    kept = [d for d in want if d in have]
    if kept != have:
        code = f'tl.permute({code}, {tuple(have.index(d) for d in kept)})'
    if not kept or len(kept) == len(want):
        return code
    return f"{code}[{', '.join(':' if d in have else 'None' for d in want)}]"


@dataclass
class _Tile:
    """ A value in a kernel: per assignment of indices to its unrolled axes, ((identity, index), ...), the source of a tile of `axes`, each (identity, class); and those `axes` it is zero along off the masks, and whether it is finite there. """
    entries: dict
    axes: list
    units: list
    zero: set
    finite: bool


def _assignments(units, length):
    """ Every assignment of indices to the unrolled axes `units`. """
    return [tuple(zip([d for d, _ in units], index)) for index in itertools.product(*(range(length[c]) for _, c in units))]


def _pick(tile, assignment):
    """ The entry of `tile` an assignment of indices to the unrolled axes of a frame, as a dict, selects. """
    return tile.entries[tuple((d, assignment[d]) for d, _ in tile.units)]


def _sorted(axes):
    return sorted(dict.fromkeys(axes), key=lambda a: (a[0] != 'r', a[0] if a[0] != 'r' else 0))


class _Printer:
    """
    The source of the kernel of a :class:`Region`. Within one binding the identity of an axis is 'r' for the rows and otherwise its position, right aligned as broadcasting aligns them,
    or its einsum letter. An axis of at most UNROLL entries is unrolled, so that a value is a tile per index along it; the others are axes of tiles.
    Every tile is zero where a mask is off if it was loaded, or made from such tiles by what keeps zero zero, so only a sum over another padded axis masks what it sums.
    """
    def __init__(self, plan, region, block, precision):
        self.plan, self.region, self.block, self.precision = plan, region, block, precision
        self.values, self.lines, self.guard, self.product = {}, [], None, 1
        self.unrolled = {c for c, n in plan.length.items() if 1 < n <= UNROLL}

    def padded(self, c):
        return c == ROWS or self.plan.block[c] != self.plan.length[c]

    def size(self, c):
        return self.block if c == ROWS else self.plan.block[c]

    def split(self, axes):
        return [a for a in axes if a[1] not in self.unrolled], [a for a in axes if a[1] in self.unrolled]

    def mask(self, axes, only=None):
        """ The mask of the padded axes of a tile of `axes`, or of those of them in `only`. """
        identities = [d for d, _ in axes]
        return ' & '.join(_fit('rm' if c == ROWS else f'm{c}', (d,), identities) for d, c in axes if self.padded(c) and (only is None or d in only)) or None

    def sum(self, code, axes, zero, over):
        """ `code`, a tile of `axes`, summed over the identities `over`, masked where it is not zero off the masks. """
        masked = {d for d, c in axes if d in over and self.padded(c) and d not in zero}
        if masked:
            code = f'tl.where({self.mask(axes, masked)}, {code}, 0.0)'
        for k in sorted((k for k, (d, _) in enumerate(axes) if d in over), reverse=True):
            code = f'tl.sum({code}, axis={k})'
        return code, [a for a in axes if a[0] not in over]

    def reduced(self, t, want):
        """ `t` summed over the axes, tile or unrolled, that `want` lacks. """
        keep = {d for d, _ in want}
        axes, units = [a for a in t.axes if a[0] in keep], [a for a in t.units if a[0] in keep]
        extra = {d for d, _ in t.axes} - keep
        entries = defaultdict(list)
        for key, code in t.entries.items():
            if extra:
                code, _ = self.sum(code, t.axes, t.zero, extra)
            entries[tuple((d, i) for d, i in key if d in keep)].append(code)
        entries = {key: codes[0] if len(codes) == 1 else f"({' + '.join(codes)})" for key, codes in entries.items()}
        return _Tile(entries, axes, units, t.zero & keep, t.finite)

    def tile(self, x, ident):
        """ The value `x` with its axes renamed by `ident`. """
        axes, units = self.split([(ident(d), c) for d, c in self.plan.axes(x)])
        entries = {tuple(zip([d for d, _ in units], key)): code for key, code in self.values[x][0].items()}
        return _Tile(entries, axes, units, {ident(d) for d in self.values[x][1]}, self.values[x][2])

    def named(self, x):
        return len(self.plan.p.shapes[x].named)

    @staticmethod
    def aligned(m, n):
        """ The identity among `n` named axes, right aligned, of an axis of a value of `m`. """
        return lambda d: d if d == 'r' else n - m + d

    def pointer(self, access, identities, assignment=()):
        terms = [access.buffer]
        index = dict(assignment)
        c0 = access.offset[0] + sum(index[q] * stride for q, c, stride in access.axes if q in index)
        terms += [str(c0)] * bool(c0) + [f'{access.offset[1]} * R'] * bool(access.offset[1])
        if access.rows is not None:
            terms.append(f"{_fit('r', ('r',), identities)} * {access.rows}")
        terms += [f"{_fit(f'i{c}', (q,), identities)}{f' * {stride}' if stride != 1 else ''}" for q, c, stride in access.axes if q not in index and c not in self.unrolled]
        return ' + '.join(terms)

    def access_axes(self, access):
        return self.split([('r', ROWS)] * (access.rows is not None) + [(q, c) for q, c, _ in access.axes])

    def emit(self, line, depth):
        self.lines.append('    ' * depth + line)
        self.guard = None

    def guarded(self, flag, depth, *lines):
        """ `lines` under `if flag:`, which consecutive lines of one flag share. """
        if flag and self.guard != (flag, depth, len(self.lines)):
            self.emit(f'if {flag}:', depth)
        for line in lines:
            self.emit(line, depth + bool(flag))
        self.guard = (flag, depth, len(self.lines)) if flag else None

    def load(self, s, access, depth):
        axes, units = self.access_axes(access)
        mask, identities, names = self.mask(axes), [d for d, _ in axes], {}
        for assignment in _assignments(units, self.plan.length):
            names[tuple(i for _, i in assignment)] = name = f"v{len(self.values)}{''.join(f'_{i}' for _, i in assignment)}"
            self.emit(f"{name} = tl.load({self.pointer(access, identities, assignment)}{f', mask={mask}, other=0.0' if mask else ''})", depth)
        self.values[s] = (names, {d for d, c in axes if self.padded(c)}, True)

    def store(self, value, access):
        axes, units = self.access_axes(access)
        mask, identities, lines = self.mask(axes), [d for d, _ in axes], []
        t = self.tile(value, self.aligned(self.named(value), access.named)) if isinstance(value, sympy.Symbol) else _Tile({(): repr(float(value))}, [], [], set(), True)
        t = self.reduced(t, axes + units)
        for assignment in _assignments(units, self.plan.length):
            code = _pick(t, dict(assignment))
            code = _fit(code, [d for d, _ in t.axes], identities) if t.axes else code
            lines.append(f"tl.store({self.pointer(access, identities, assignment)}, {code}{f', mask={mask}' if mask else ''})")
        return lines

    def expression(self, e, n):
        """ The pointwise `e` among `n` named axes. """
        if isinstance(e, sympy.Symbol):
            if e in self.plan.p.lengths:
                return _Tile({(): ' * '.join('R' if c == ROWS else str(self.plan.length[c]) for c in self.plan.p.lengths[e]) or '1'}, [], [], set(), True)
            return self.tile(e, self.aligned(self.named(e), n))
        if e.is_number:
            return _Tile({(): repr(float(e))}, [], [], set(), True)
        parts = [self.expression(a, n) for a in (e.args[:1] if isinstance(e, sympy.Pow) else e.args)]
        axes, units = _sorted(a for t in parts for a in t.axes), _sorted(a for t in parts for a in t.units)
        want, finite = [d for d, _ in axes], all(t.finite for t in parts)
        if isinstance(e, sympy.Add):
            zero = set.intersection(*(t.zero for t in parts))
        elif isinstance(e, sympy.Mul):
            zero = set.union(*(t.zero for t in parts)) if finite else set()
        else:
            zero = parts[0].zero
        entries = {}
        for assignment in _assignments(units, self.plan.length):
            codes = [_fit(_pick(t, dict(assignment)), [d for d, _ in t.axes], want) if t.axes else _pick(t, dict(assignment)) for t in parts]
            if isinstance(e, sympy.Add):
                code = f"({' + '.join(codes)})"
            elif isinstance(e, sympy.Mul):
                code = f"({' * '.join(codes)})"
            elif isinstance(e, sympy.Pow):
                q = sympy.Rational(str(e.exp))
                root = codes[0] if q.q == 1 else f'tl.sqrt({codes[0]})' if q.q == 2 else f'tl.sqrt(tl.sqrt({codes[0]}))'
                product = f"({' * '.join([root] * abs(q.p))})" if q.p else '1.0'
                code = product if q > 0 else f'(1.0 / {product})'
            else:
                a = codes[0]
                code = f'tl.where({a} > 0, 1.0, tl.where({a} < 0, -1.0, 0.0))' if e.func.__name__ == 'sign' else f'{FUNCTIONS[e.func.__name__]}({a})'
            entries[assignment] = code
        if isinstance(e, sympy.Pow) and sympy.Rational(str(e.exp)) < 0:
            zero, finite = set(), False
        elif not isinstance(e, (sympy.Add, sympy.Mul, sympy.Pow)):
            name = e.func.__name__
            zero, finite = zero if name in ODD else set(), finite and name in BOUNDED
        return _Tile(entries, axes, units, zero, finite)

    def product_of(self, codes, output, contracted, zero, finite):
        """ The product of the tiles `codes`, each (source, axes), summed over the tile axes `contracted` into `output`: a tl.dot where it is one product of two tiles at least that large along each axis. """
        if len(codes) == 2 and len(contracted) == 1 and len(output) == 2 and all(self.size(c) >= DOT for _, c in (*output, *contracted)):
            (k, kc), (m, _), (n, _) = contracted[0], *output
            for (a, xa, za, fa), (b, xb, zb, fb) in (codes, codes[::-1]):
                if {d for d, _ in xa} == {m, k} and {d for d, _ in xb} == {k, n}:
                    if self.padded(kc) and not (k in za and fb or k in zb and fa):
                        a = f'tl.where({self.mask(xa, {k})}, {a}, 0.0)'
                    return f'tl.dot({_fit(a, [d for d, _ in xa], (m, k))}, {_fit(b, [d for d, _ in xb], (k, n))}, input_precision={self.precision!r})', output
        union = [a for a in contracted if a[1] == ROWS] + output + [a for a in contracted if a[1] != ROWS]
        if contracted and any(c == ROWS for _, c in union):
            self.product = max(self.product, math.prod(self.size(c) for _, c in union if c != ROWS))
        identities = [d for d, _ in union]
        product = ' * '.join(_fit(c, [d for d, _ in x], identities) if x else c for c, x, _, _ in codes)
        return self.sum(f'({product})', union, zero, {d for d, _ in contracted})

    def contraction(self, e, rows, ident):
        """ The Einsum `e`, whose output has the rows if `rows`, with its axes renamed by `ident`; an unrolled axis it contracts is a sum of products. """
        letters, out = _letters(e)
        tiles = [self.tile(o, lambda d, term=term: '...' if d == 'r' else term[d]) for o, term in zip(e.args[1:], letters)]
        have, units = _sorted(a for t in tiles for a in t.axes), list(dict.fromkeys(a for t in tiles for a in t.units))
        output = [a for a in have if a[0] == '...'] * rows + [a for t in out for a in have if a[0] == t]
        kept = [a for t in out for a in units if a[0] == t]
        contracted, summed = [a for a in have if a not in output], [a for a in units if a not in kept]
        finite = all(t.finite for t in tiles)
        zero = set.union(*(t.zero for t in tiles)) if finite else set()
        entries, axes = {}, output
        for assignment in _assignments(kept, self.plan.length):
            terms = []
            for inner in _assignments(summed, self.plan.length):
                index = dict(assignment + inner)
                code, axes = self.product_of([(_pick(t, index), t.axes, t.zero, t.finite) for t in tiles], output, contracted, zero, finite)
                terms.append(code)
            entries[tuple((ident(d), i) for d, i in assignment)] = terms[0] if len(terms) == 1 else f"({' + '.join(terms)})"
        return _Tile(entries, [(ident(d), c) for d, c in axes], [(ident(d), c) for d, c in kept], {ident(d) for d in zero if d in dict(axes)}, finite)

    def binding(self, s, e, depth):
        if isinstance(e, Reshape):
            self.values[s] = self.values[e.args[0]]
            return
        want, n = self.plan.axes(s), self.named(s)
        if isinstance(e, Einsum):
            _, out = _letters(e)
            t = self.contraction(e, self.plan.full[s][0], lambda d: 'r' if d == '...' else out.index(d))
        else:
            m = max([self.named(x) for x in _operands(e) if x in self.plan.p.shapes and x not in self.plan.p.lengths] + [0])
            t = self.expression(e, m)
            shift = self.aligned(m, n)
            t = _Tile({tuple((shift(d), i) for d, i in key): code for key, code in t.entries.items()}, [(shift(d), c) for d, c in t.axes], [(shift(d), c) for d, c in t.units],
                      {shift(d) for d in t.zero}, t.finite)
            if isinstance(e, sympy.Symbol) and {d for d, _ in t.axes + t.units} == {d for d, _ in want}:
                self.values[s] = self.values[e]
                return
        t = self.reduced(t, want)
        tiles, units = self.split(want)
        names, identities = {}, [d for d, _ in tiles]
        number = len(self.values)
        for assignment in _assignments(units, self.plan.length):
            key = tuple(i for _, i in assignment)
            names[key] = name = f"v{number}{''.join(f'_{i}' for i in key)}"
            code = _pick(t, dict(assignment))
            self.emit(f'{name} = {_fit(code, [d for d, _ in t.axes], identities) if t.axes else code}', depth)
        self.values[s] = (names, t.zero & set(identities), t.finite)

    def dotted(self, a):
        """ Which terms of the accumulator `a` a tl.dot sums over the rows, as :meth:`product_of` would; the others go, block by block, into a tile that keeps them, summed once after the last. """
        flags = []
        for _, e, _ in a.terms:
            letters, out = _letters(e)
            tiles = [self.split([('...' if d == 'r' else term[d], c) for d, c in self.plan.axes(o)])[0] for o, term in zip(e.args[1:], letters)]
            have = list(dict.fromkeys(x for t in tiles for x in t))
            output = [x for t in out for x in have if x[0] == t]
            contracted = [x for x in have if x not in output]
            flags.append(len(tiles) == 2 and len(contracted) == 1 and len(output) == 2 and all(self.size(c) >= DOT for _, c in have)
                         and {frozenset(tiles[0]), frozenset(tiles[1])} == {frozenset([output[0], contracted[0]]), frozenset([contracted[0], output[1]])})
        return flags

    def accumulate(self, a, terms, depth):
        """ The block's contribution of each term of the accumulator `a`; a reduction several accumulators share is computed once, into `terms`. """
        axes, units = self.access_axes(a.access)
        n, identities = a.access.named, [d for d, _ in axes]
        for (c, e, reduction), dotted in zip(a.terms, self.dotted(a)):
            want = axes if dotted else [('r', ROWS)] + axes
            if (reduction, dotted) not in terms:
                _, out = _letters(e)
                t = self.reduced(self.contraction(e, not dotted, lambda d: 'r' if d == '...' else n - len(out) + out.index(d)), want + units)
                number = len(terms)
                terms[reduction, dotted] = t
                for key, code in t.entries.items():
                    if not dotted and 'r' not in t.zero:
                        code = f"tl.where({_fit('rm', ('r',), [d for d, _ in t.axes])}, {code}, 0.0)"
                    name = f"t{number}{''.join(f'_{i}' for _, i in key)}"
                    self.guarded(a.flag, depth, f'{name} = {code}')
                    t.entries[key] = name
            t = terms[reduction, dotted]
            coefficient = self.expression(c, n) if c != 1 else None
            ids = [d for d, _ in want]
            for assignment in _assignments(units, self.plan.length):
                term = _pick(t, dict(assignment))
                term = _fit(term, [d for d, _ in t.axes], ids) if t.axes else term
                if coefficient is not None:
                    code = _pick(coefficient, dict(assignment))
                    term = f"{_fit(code, [d for d, _ in coefficient.axes], ids) if coefficient.axes else code} * {term}"
                self.guarded(a.flag, depth, f"{self.accumulators[id(a)][dotted][tuple(i for _, i in assignment)]} += {term}")

    def source(self, name):
        r, plan = self.region, self.plan
        flags = [f'{f}: tl.constexpr' for f in r.flags]
        pointers = [*r.reads, *(b.name for b in r.writes)]
        self.emit(f"def {name}({', '.join([*pointers, 'R', *flags])}):", 0)
        self.emit('pid = tl.program_id(0)', 1)
        self.emit(f'DT = {pointers[0]}.dtype.element_ty', 1)
        accesses = [a for _, a, _ in r.stores] + [a.access for a in r.accumulators]
        classes = sorted(({c for x in (*r.loads, *(x for x, _ in r.bindings)) for _, c in plan.axes(x)} | {c for a in accesses for _, c, _ in a.axes}
                          | {c for a in r.accumulators for _, e, _ in a.terms for o in e.args[1:] for _, c in plan.axes(o)}) - {ROWS} - self.unrolled)
        for c in classes:
            self.emit(f'i{c} = tl.arange(0, {plan.block[c]})', 1)
            if plan.block[c] != plan.length[c]:
                self.emit(f'm{c} = i{c} < {plan.length[c]}', 1)
        for s, a in r.loads.items():
            if not plan.full[s][0]:
                self.load(s, a, 1)
        for s, e in r.bindings:
            if not plan.full[s][0]:
                self.binding(s, e, 1)
        self.accumulators = {}
        for a in r.accumulators:
            axes, units = self.access_axes(a.access)
            number, dotted = len(self.accumulators), self.dotted(a)
            self.accumulators[id(a)] = {True: {}, False: {}}
            for kind in {*dotted} | ({True} if not a.terms else set()):
                for assignment in _assignments(units, plan.length):
                    key = tuple(i for _, i in assignment)
                    acc = self.accumulators[id(a)][kind][key] = f"acc{number}{'' if kind else 'r'}{''.join(f'_{i}' for i in key)}"
                    self.emit(f"acc{number}{'' if kind else 'r'}{''.join(f'_{i}' for i in key)} = tl.zeros({[self.size(c) for _, c in ([] if kind else [('r', ROWS)]) + axes]}, DT)", 1)
        if r.rows:
            self.emit(f'for blk in range(pid, tl.cdiv(R, {self.block}), tl.num_programs(0)):', 1)
            self.emit(f'r = blk * {self.block} + tl.arange(0, {self.block})', 2)
            self.emit('rm = r < R', 2)
            for s, a in r.loads.items():
                if plan.full[s][0]:
                    self.load(s, a, 2)
            for s, e in r.bindings:
                if plan.full[s][0]:
                    self.binding(s, e, 2)
            for value, a, flag in r.stores:
                if a.rows is not None:
                    self.guarded(flag, 2, *self.store(value, a))
            terms = {}
            for a in r.accumulators:
                self.accumulate(a, terms, 2)
        for value, a, flag in r.stores:
            if a.rows is None:
                self.guarded(flag, 1, 'if pid == 0:', *('    ' + line for line in self.store(value, a)))
        for a in r.accumulators:
            axes, units = self.access_axes(a.access)
            identities = [d for d, _ in axes]
            rest = self.reduced(self.expression(a.rest, a.access.named), axes + units) if a.rest != 0 else None
            numel = math.prod(next(b for b in r.writes if b.name == a.access.buffer).shape)
            mask = self.mask(axes)
            for assignment in _assignments(units, plan.length):
                key = tuple(i for _, i in assignment)
                held = self.accumulators[id(a)]
                parts = ([held[True][key]] if key in held[True] else []) + ([f"tl.sum({held[False][key]}, axis=0)"] if key in held[False] else [])
                acc = f"acc{list(self.accumulators).index(id(a))}{''.join(f'_{i}' for i in key)}"
                lines = [f"{acc} = {' + '.join(parts)}"] if parts != [acc] else []
                if rest is not None:
                    code = _pick(rest, dict(assignment))
                    lines += ['if pid == 0:', f"    {acc} += {_fit(code, [d for d, _ in rest.axes], identities) if rest.axes else code}"]
                self.guarded(a.flag, 1, *lines, f"tl.store({self.pointer(a.access, identities, assignment)} + pid * {numel}, {acc}{f', mask={mask}' if mask else ''})")
        return '\n'.join(self.lines) + '\n'


def _interpreting():
    return os.environ.get('TRITON_INTERPRET') == '1'


def _before_symbolic():
    import torch
    if torch.compiler.is_compiling():
        raise Unsupported('triton2 prepares an operator in a regular call before torch.compile traces it')


class _Variant:
    """ A compiled expression for operands of one variant: the kernel of every region as a custom op, and the autograd.Function that runs its steps. """
    def __init__(self, prepared, values):
        import torch

        self.p = prepared
        float32 = values[0].dtype == torch.float32
        precision = {'highest': 'ieee', 'high': 'tf32x3', 'medium': 'tf32'}[torch.get_float32_matmul_precision()] if float32 else 'ieee'
        floor = lambda n: 1 << (max(n, 1).bit_length() - 1)
        # Only Triton knows the shared memory a kernel with a tl.dot needs: such a kernel is compiled before the variant runs, and where it needs more than the card has,
        # its pass keeps half the weights and gives the rest to Torch.
        budgets = [SHARED, SHARED]
        while True:
            plan = self.plan = _Plan(prepared, values, values[0].element_size(), budgets)
            self.forward, self.backward, self.saves = plan.plan()
            regions = [r for r in (*self.forward, *self.backward) if isinstance(r, Region)]
            self.blocks, self.warps, self.kernels, self.ops, self.caps, self.pending, sources = {}, {}, {}, {}, {}, {}, {}
            for r in regions:
                printer = _Printer(plan, r, DOT, precision)
                sources[r.name] = printer.source(r.name)
                self.blocks[r.name], self.warps[r.name] = DOT, 2 * WARPS
                if 'tl.dot' not in sources[r.name]:
                    widest = max((math.prod(plan.block[c] for c in plan.dims(s) if c not in printer.unrolled and c != ROWS) for s in [*r.loads, *(s for s, _ in r.bindings)] if plan.full[s][0]), default=1)
                    threads = 32 * WARPS
                    self.blocks[r.name], self.warps[r.name] = min(floor(VALUES * threads // widest), floor(PRODUCTS * threads // printer.product)), WARPS
                    sources[r.name] = self.source(r, self.blocks[r.name], precision)
            self.compile(regions, sources, values, precision)
            over = {r.name for r in regions if 'tl.dot' in sources[r.name] and not self.admitted(r.name)}
            if not over:
                break
            for k, steps in enumerate((self.forward, self.backward)):
                if any(isinstance(r, Region) and r.name in over for r in steps):
                    if not budgets[k]:
                        raise Unsupported(f'{prepared.name}, whose kernels need more shared memory than the card has even without weights')
                    budgets[k] //= 2
        for r in regions:
            self.ops[r.name] = self.custom_op(r)
        variant = self

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, save, *values):
                env, R, ell = variant.environment(values)
                variant.run(variant.forward, env, R, {'SAVE': save})
                if save:
                    ctx.save_for_backward(*values, *(env[name] for name in variant.saves))
                return env['out'].view(variant.out_shape(ell))

            @staticmethod
            def backward(ctx, gout):
                saved = ctx.saved_tensors
                values = saved[:len(variant.p.args)]
                env, R, _ = variant.environment(values)
                env.update(zip(variant.saves, saved[len(values):]))
                env['gout'] = gout.contiguous()
                flags = {f'G{i}': ctx.needs_input_grad[i + 1] for i in range(len(values))}
                variant.run(variant.backward, env, R, flags)
                return None, *(env[f'd{i}'].view(v.shape) if flags[f'G{i}'] and f'd{i}' in env else None for i, v in enumerate(values))

        self.function = Function

    def source(self, region, block, precision):
        return _Printer(self.plan, region, block, precision).source(region.name)

    def compile(self, regions, sources, values, precision):
        """
        Every kernel, compiled on a card in the background and waited for at its first launch, so that a backward compiles while the forward runs
        and the operators after it are prepared.
        """
        import triton
        import triton.language as tl

        jit = lambda r: triton.jit(do_not_specialize=['R'])(_compile_and_cache(sources[r.name], r.name, {'tl': tl, 'triton': triton}))
        self.kernels = {r.name: jit(r) for r in regions}
        if _interpreting():
            self.caps = dict.fromkeys(self.kernels, 2)
            return
        import torch
        dtype, properties = values[0].dtype, torch.cuda.get_device_properties(values[0].device)

        def compiled(r):
            kernel = self.kernels[r.name].warmup(*[dtype] * (len(r.reads) + len(r.writes)), 1024, grid=(1,), num_warps=self.warps[r.name], num_stages=STAGES, **dict.fromkeys(r.flags, True))
            kernel._init_handles()  # Triton reads the registers, and checks the shared memory, only when it loads the binary.
            threads, shared = 32 * self.warps[r.name], kernel.metadata.shared
            resident = min(properties.max_threads_per_multi_processor // threads, properties.regs_per_multiprocessor // max(kernel.n_regs * threads, 1),
                           properties.shared_memory_per_multiprocessor // shared if shared else threads)
            self.caps[r.name] = properties.multi_processor_count * max(resident, 1)

        self.pending = {r.name: _pool().submit(compiled, r) for r in regions}

    def admitted(self, name):
        """ Whether the kernel `name` compiles within the card's shared memory, which this waits for. """
        from triton.runtime.errors import OutOfResources
        try:
            if (compiling := self.pending.pop(name, None)) is not None:
                compiling.result()
        except OutOfResources:
            return False
        return True

    def custom_op(self, r):
        import torch

        warps = self.warps[r.name]

        def allocate(like, R, flags, programs=None):
            out = []
            for b in r.writes:
                if b.flag and not flags.get(b.flag, True):
                    out.append(like.new_empty(0))
                    continue
                shape = list(b.shape)
                if b.rows is not None:
                    shape.insert(b.rows, R)
                if b.partial and programs is not None:
                    out.append(like.new_zeros((programs, *shape)))
                else:
                    out.append(like.new_empty(shape))
            return out

        def run(inputs, R, flags):
            if (compiling := self.pending.pop(r.name, None)) is not None:
                compiling.result()
            kernel, block = self.kernels[r.name], self.blocks[r.name]
            flags = dict(zip(r.flags, flags))
            blocks = -(-R // block) if r.rows else 1
            # Every program loops over blocks of rows, so that it loads what has no rows, and stores the partials of its sums, once.
            programs = min(blocks, self.caps[r.name])
            out = allocate(inputs[0], R, flags, programs)
            if programs:
                kernel[(programs,)](*inputs, *out, R, num_warps=warps, num_stages=STAGES, **flags)
            return [t.sum(0) if b.partial and not (b.flag and not flags[b.flag]) else t for b, t in zip(r.writes, out)]

        def fake(inputs, R, flags):
            return allocate(inputs[0], R, dict(zip(r.flags, flags)))

        op = torch.library.custom_op(f'kingdon_triton2::{r.name}_{next(_OPS)}', run, mutates_args=(), schema='(Tensor[] inputs, SymInt R, bool[] flags) -> Tensor[]')
        op.register_fake(fake)
        return op, run

    def environment(self, values):
        """ The operands by name, their rows R, and the batch dimensions those are. """
        env = {f'a{i}': v for i, v in enumerate(values)}
        for arg, (lead, named) in enumerate(self.p.layouts):
            if named is not None:
                v = values[arg]
                ell = tuple(v.shape[lead:v.ndim - named])
                return env, math.prod(ell), ell
        return env, 1, ()

    def out_shape(self, ell):
        n = len(self.p.outputs) // math.prod(self.p.entries)
        return (n, *self.p.entries, *(ell if self.plan.out_rows else ()), *self.plan.out_lengths)

    def run(self, steps, env, R, flags):
        import torch
        for step in steps:
            if isinstance(step, TorchCall):
                if step.flag and not flags.get(step.flag, True):
                    continue
                operands = [env[buffer][index].view(*(R if d is None else d for d in shape)) for buffer, index, shape in step.operands]
                env[step.result.name] = torch.einsum(step.pattern, *operands).contiguous()
            else:
                # The custom op keeps a tracing compiler out of the kernel; outside one, its dispatch only costs time.
                op, run = self.ops[step.name]
                outputs = (op if torch.compiler.is_compiling() else run)([env[name] for name in step.reads], R, [flags.get(f, True) for f in step.flags])
                env.update(zip((b.name for b in step.writes), outputs))

    def __call__(self, *values):
        import torch
        values = tuple(v.contiguous() for v in values)
        save = torch.is_grad_enabled() and any(v.requires_grad for v in values)
        return self.function.apply(save, *values)


_OPS = itertools.count()
_POOL = []


def _pool():
    """ The threads every variant compiles its kernels in: Triton releases the GIL for most of a compile. """
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor
        _POOL.append(ThreadPoolExecutor(os.cpu_count()))
    return _POOL[0]


def _key(prepared, values):
    """ What a variant is for: per operand its dtype, device, number of axes and their lengths but for the rows. The matmul precision is read when a variant is made, since Dynamo cannot trace it. """
    import torch
    key = []
    for v, (lead, named) in zip(values, prepared.layouts):
        if not isinstance(v, torch.Tensor):
            raise Unsupported(f'coefficients of type {type(v).__name__}, where triton2 takes one tensor per operand')
        if not lead:  # An operand none of whose coefficients is used.
            key.append((v.dtype, v.device))
            continue
        shape = v.shape if named is None else (*v.shape[:lead], *(v.shape[v.ndim - named:] if named else ()))
        key.append((v.dtype, v.device, v.ndim, *map(int, shape)))
    return tuple(key)


def _check(values):
    """ Kernels run natively on CUDA float32 tensors, and in Triton's interpreter on the CPU. """
    for v in values:
        if _interpreting() and v.device.type == 'cpu' and v.dtype.is_floating_point:
            continue
        if not (v.is_cuda and v.dtype.is_floating_point and v.dtype.itemsize == 4):
            raise Unsupported(f'operands of {v.dtype} on {v.device}: triton2 runs natively on CUDA float32 tensors, and on the CPU under TRITON_INTERPRET=1')


def triton_lambdify(args, exprs, funcname, cse=True, output_mv_idx=None, shapes=None, keys=None, codegen_symbolcls=None, wrapper=None):
    """
    The numerical function of a compiled expression, backed by Triton kernels: see :doc:`backends/triton2`.
    Its symbolic work happens here, once; each new variant of its operands, widths say, gets its kernels at its first call, outside torch.compile.
    `wrapper` is not applied: the kernels are custom ops already, which leave a compiler of the caller nothing to do.
    """
    if output_mv_idx is not None:
        raise Unsupported(f'{funcname} writes into one of its arguments')
    prepared = _prepare(args, exprs, funcname, cse, shapes, codegen_symbolcls)
    variants = {}

    def numerical(*values):
        import torch
        compiling = torch.compiler.is_compiling()
        if compiling:  # Dynamo takes no tensor twice as the argument of an autograd.Function.
            values = tuple(v.view_as(v) if any(v is u for u in values[:i]) else v for i, v in enumerate(values))
        key = _key(prepared, values)
        if (variant := variants.get(key)) is None:
            if compiling:
                raise Unsupported(f'{funcname} has no kernels yet for operands of {key}: call it once with them before torch.compile traces it')
            _check(values)
            variant = variants[key] = _Variant(prepared, values)
        return variant(*values)

    numerical.__name__ = funcname
    return numerical


triton_lambdify.before_symbolic = _before_symbolic
