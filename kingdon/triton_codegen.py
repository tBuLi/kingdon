"""
One Triton kernel per operator, instead of one torch call per symbolic multiply::

    >>> alg = Algebra(3, lambdifier=triton_lambdify)
    >>> z = alg.gp(x, y)

Multivector in, multivector out, differentiable, same as any other lambdifier. The expressions
are kingdon's polynomials or, for an operator whose codegen_symbolcls is a sympy symbol, sympy's,
which may then call the functions triton has, :code:`erf` say. Anything else falls back to
:func:`~kingdon.codegen.lambdify`. The backward is the expressions' own derivative, taken symbolically.
"""
from __future__ import annotations

import bisect
import functools
import itertools
import linecache
import math
from dataclasses import dataclass, replace

import numpy as np
import sympy
from sympy.core.symbol import Str
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import PythonCodePrinter

from kingdon.codegen import ArrayBase, Blade, Einsum, Reduce, Reshape, Stack, _bottom_up
from kingdon.polynomial import RationalPolynomial, poly_format, rational_cse, rp_var_name

#: Tiles to consider, as (elements, warps, stages). How many registers a tile needs is
#: not worth predicting -- measured against a count of the live coefficients the ratio ranged
#: from 0.92 to 2.09 -- so :func:`_widest_clean` compiles them and reads the spills instead.
#: :func:`_fit` shapes each of these to the plane before any of that.
CONFIGS = [(64, 8, 1), (64, 4, 1), (128, 4, 1), (512, 16, 1), (1024, 16, 2)]

_BUILDS = itertools.count()


class Unsupported(Exception):
    """Raised when an expression cannot be emitted as a kernel, so the caller falls back."""


#: The name of the row axis of a tile in an einsum pattern, where einops names axes with letters.
_ROW = '.'
#: What the padding of a tile holds for a reduction to ignore it.
_IDENTITY = {'max': 'float("-inf")', 'min': 'float("inf")'}


class TritonPrinter(PythonCodePrinter):
    """
    Sympy expressions as triton code: numbers as floats, the functions triton has as its own, and powers as the products and roots it can take.
    Given the tile of every symbol, as ``(rows, features)`` -- whether it has the row axis, and the sizes of the axes after it -- it prints einops nodes on tiles too.
    A node over multivectors, a Stack of blades, is printed one blade at a time, see :meth:`_blade`, but where it sums over the blades.
    """

    def __init__(self, shapes=None, precision='ieee'):
        super().__init__({'user_functions': {name: f'tl.{name}' for name in ('erf', 'exp', 'log', 'sin', 'cos', 'sigmoid')}})
        self.shapes, self.precision, self.counts = shapes, precision, {}
        # The symbols loaded with zeros in their padding, which a contraction need not zero again.
        self.zeroed = set()
        self.shape = functools.cache(self._shape)

    def count(self, e):
        """How many blades `e` holds as a tuple of tiles, or None for a tile."""
        if isinstance(e, Stack):
            return len(e.args)
        if isinstance(e, Einsum):
            counts = [self.count(op) for op in e.args[1:]]
            return next(filter(None, counts)) if _unbladed(e, counts)[1] else None
        if isinstance(e, (Reduce, Reshape)):
            return self.count(e.args[0])
        return self.counts.get(e)

    def _print(self, expr, **kwargs):
        if isinstance(expr, sympy.Basic) and expr.is_number and not expr.is_Integer:
            return repr(float(expr))
        return super()._print(expr, **kwargs)

    def _print_Pow(self, expr, rational=False):
        if not expr.exp.is_number or (power := sympy.Rational(expr.exp)).q not in (1, 2, 4):
            raise Unsupported(f'power {expr.exp}')
        roots = power.q.bit_length() - 1
        base = self._print(expr.base) if roots else self.parenthesize(expr.base, PRECEDENCE['Pow'])
        for k in range(roots):
            base = f'tl.{"rsqrt" if power < 0 and k == roots - 1 else "sqrt"}({base})'
        product = '*'.join([base] * abs(power.p))
        # Parenthesized, since the printer takes a power for an atom: a/x**2 would print as a/x*x.
        return f'(1/({product}))' if power < 0 and not roots else f'({product})' if abs(power.p) > 1 else product

    def _shape(self, e):
        """The tile of `e`, or of each of its blades."""
        if isinstance(e, sympy.Symbol):
            return self.shapes[e]
        if isinstance(e, Blade):
            return self.shape(e.args[0])
        if isinstance(e, Einsum):
            _, _, terms, right = _unbladed(e, [self.count(op) for op in e.args[1:]])
            tiles = [self.shape(op) for op in e.args[1:]]
            sizes = {letter: n for term, (_, features) in zip(terms, tiles) for letter, n in zip(term.replace('...', ''), features)}
            return '...' in right and any(rows for term, (rows, _) in zip(terms, tiles) if '...' in term), tuple(sizes[letter] for letter in right.replace('...', ''))
        if isinstance(e, Reduce):
            rows, features = self.shape(e.args[0])
            return rows and -1 - len(features) not in e.args[2], tuple(n for k, n in enumerate(features) if k - len(features) not in e.args[2])
        if isinstance(e, Reshape):
            (rows, features), k, sizes = self.shape(e.args[0]), int(e.args[1]), tuple(map(int, e.args[2]))
            if k > len(features) + rows:
                raise Unsupported('a reshape across rows')
            if k > len(features):  # The -1 is the row axis.
                return rows, sizes[1:]
            flat = math.prod(features[len(features) - k:]) // math.prod(n for n in sizes if n != -1)
            return rows, features[:len(features) - k] + tuple(flat if n == -1 else n for n in sizes)
        tiles = [self.shape(a) for a in e.args]
        return any(rows for rows, _ in tiles), tuple(np.broadcast_shapes(*(features for _, features in tiles)))

    def _blade(self, e, i):
        """Blade i of `e`, an einops node over multivectors, as the same node over its tiles."""
        pick = lambda op: op.args[i] if isinstance(op, Stack) else Blade(op, i)
        if not isinstance(e, Einsum):
            return e.func(pick(e.args[0]), *e.args[1:])
        counts = [self.count(op) for op in e.args[1:]]
        _, _, terms, right = _unbladed(e, counts)
        return Einsum(Str(f'{",".join(terms)}->{right}'), *(pick(op) if n else op for op, n in zip(e.args[1:], counts)))

    def _print_Einsum(self, e):
        """Over tiles, one contraction: a node over multivectors, or their blades, is one over tiles by the time it is printed, see :func:`_emit`."""
        blade, _, terms, right = _unbladed(e, [self.count(op) for op in e.args[1:]])
        if blade is not None:
            raise Unsupported('a multivector as one value')
        return self._contract(terms, right, e.args[1:])

    def _contract(self, terms, right, ops):
        """In registers: a tl.dot where the pattern is a product of matrices large enough for one, else the operands broadcast against each other, multiplied and summed."""
        zeroed = [op in self.zeroed for op in ops]
        ops = [(self._print(op), self.shape(op)) for op in ops]
        names = [(_ROW if rows and '...' in term else '') + term.replace('...', '') for term, (_, (rows, _)) in zip(terms, ops)]
        out = (_ROW if '...' in right and any(rows for term, (_, (rows, _)) in zip(terms, ops) if '...' in term) else '') + right.replace('...', '')
        summed = [letter for letter in dict.fromkeys(''.join(names)) if letter not in out]
        axes = {letter: axis for name, (_, tile) in zip(names, ops) for letter, axis in zip(name, _axes(tile))}
        pads = [_pad(n) for _, (_, features) in ops for n in features]

        def masked(text, name, zero=False):
            """`text`, a tile with axes `name`, with zeros in the padding of the summed axes, unless it has those already: a padded lane may hold anything after a division or a root."""
            masks = [_expand(axes[letter][1], name.index(letter), len(name)) for letter in summed if letter in name and axes[letter][1]]
            return f'tl.where({" & ".join(masks)}, {text}, 0.0)' if masks and not zero else text

        if len(ops) == 2 and len(summed) == 1 and len(out) == 2 and all(len(name) == 2 for name in names) and min(pads) >= 16:
            k = summed[0]
            (a, _), (b, _) = ops
            m, n = names[0].replace(k, ''), names[1].replace(k, '')
            if m != n:
                a, b = masked(a, names[0], zeroed[0]), masked(b, names[1], zeroed[1])
                a, b = a if names[0] == m + k else f'tl.trans({a})', b if names[1] == k + n else f'tl.trans({b})'
                if out == m + n:
                    return f'tl.dot({a}, {b}, input_precision="{self.precision}")'
                return f'tl.trans(tl.dot({a}, {b}, input_precision="{self.precision}"))'

        letters = out + ''.join(summed)

        def placed(text, name):
            order = sorted(name, key=letters.index)
            if list(name) != order:
                text = f'tl.permute({text}, {tuple(name.index(letter) for letter in order)})'
            return text if len(name) == len(letters) else f'{text}[{", ".join(":" if letter in name else "None" for letter in letters)}]'

        product = masked(' * '.join(placed(text, name) for (text, _), name in zip(ops, names)), letters, all(zeroed))
        for letter in reversed(summed):
            product = f'tl.sum({product}, {letters.index(letter)})'
        return product

    def _print_Reduce(self, e):
        a, operation, axes = e.args
        rows, features = tile = self.shape(a)
        n = rows + len(features)
        ks = sorted(n + int(k) for k in axes)
        if operation.name not in ('sum', 'mean', 'max', 'min') or ks[0] < rows and operation.name != 'sum':
            raise Unsupported(f'{operation.name} over {axes}')
        text = self._print(a)
        if masks := [_expand(mask, k, n) for k in ks if (mask := _axes(tile)[k][1])]:
            text = f'tl.where({" & ".join(masks)}, {text}, {_IDENTITY.get(operation.name, "0.0")})'
        for k in reversed(ks):
            text = f'tl.{"sum" if operation.name == "mean" else operation.name}({text}, {k})'
        return f'({text}) / {math.prod(features[k - rows] for k in ks)}' if operation.name == 'mean' else text

    def _print_Reshape(self, e):
        (rows, features), target = self.shape(e.args[0]), self.shape(e)
        if (rows, [_pad(n) for n in features if n != 1]) != (target[0], [_pad(n) for n in target[1] if n != 1]):
            raise Unsupported('a reshape of more than unit axes')
        return f'tl.reshape({self._print(e.args[0])}, {_padded(target)})'


@dataclass(frozen=True)
class Operand:
    """One argument of an operator: its coefficients, and how the kernel addresses them."""

    name: str
    vars: tuple[str, ...]
    nested: bool
    #: Which axes of the coalesced plane the coefficients vary along, see :func:`_layout`.
    varies: tuple[bool, ...] | None = None
    #: The axes the coefficients carry past the blade axis, or ``None`` for a plain number.
    shape: tuple[int, ...] | None = None

    @property
    def slots(self):
        return len(self.vars)

    @property
    def lead(self):
        """How many axes precede the data: the blade axis, and a depth axis when stacked."""
        return 2 if self.nested else 1

    @property
    def array(self):
        return self.shape is not None

    @property
    def live(self):
        return [(i, var) for i, var in enumerate(self.vars) if var != '_']


def _pad(n):
    return 1 << (n - 1).bit_length()


def _padded(tile):
    rows, features = tile
    return f'[{", ".join(["T0"] * rows + [str(_pad(n)) for n in features])}]'


def _axes(tile):
    """Per axis of a tile its index and its mask, or None where no lane is padding."""
    rows, features = tile
    return [('_x0', '_m0')] * rows + [(f'tl.arange(0, {_pad(n)})', None if _pad(n) == n else f'(tl.arange(0, {_pad(n)}) < {n})') for n in features]


def _expand(text, k, n):
    """`text`, a vector, as axis k of a tile of n axes."""
    return text if n < 2 else f'{text}[{", ".join(":" if j == k else "None" for j in range(n))}]'


def _terms(einsum):
    """The operand terms and the output term of the pattern of an Einsum node."""
    lefts, right = einsum.args[0].name.split('->')
    return lefts.split(','), right


def _unbladed(einsum, mvs):
    """
    The blade axis of an Einsum node, the letter the terms of its multivector operands -- those `mvs` flags -- start with, or None; whether its output keeps that axis;
    and its operand terms and output term without it.
    """
    terms, right = _terms(einsum)
    blade = next((term[0] for term, mv in zip(terms, mvs) if mv), None)
    kept = blade is not None and right[:1] == blade
    return blade, kept, [term[1:] if mv else term for term, mv in zip(terms, mvs)], right[1:] if kept else right


def _load(tile):
    return math.prod(tile[0]) / tile[1]


def _fit(tile, widths, outer=1):
    """
    Shape a tile to the plane: the innermost axis takes its width while the elements last, the outermost up to `outer` of them, the axes between their widths from the inside out, and the outermost the rest.

    Filling from the inside keeps the lanes on real data -- a 64 wide row over the three columns of o3's first layer masks off 61 of them.
    Keeping the count keeps both the register pressure :func:`_widest_clean` measures and the elements per warp :func:`_load` orders by.
    A larger `outer` narrows the tile along the axes between, which only the backward wants: an operand shared along one of those, like the input to a fully connected product,
    has every element of the tile along it add into the same coefficient, and those adds wait on each other.
    """
    elements, warps, stages = tile
    inner = [min(elements, widths[-1])] if widths else []
    elements //= math.prod(inner)
    outer = min(outer, elements)
    elements //= outer
    shape = []
    for width in reversed(widths[:-1]):
        shape.insert(0, min(elements, width))
        elements //= shape[0]
    return (outer * elements, *shape, *inner), warps, stages


def _widths(extents):
    """All the tiles depend on: every axis but the outermost, rounded up to a power of two no larger than the largest tile."""
    most = max(elements for elements, _, _ in CONFIGS)
    return tuple(min(most, 1 << (extent - 1).bit_length()) for extent in extents[1:])


def _tiles(widths, tiles, outers=(1,)):
    """`tiles` shaped to the plane once per outer width in `outers`, less the duplicates the shaping creates."""
    return list(dict.fromkeys(_fit(tile, widths, outer) for tile in tiles for outer in outers))


@functools.cache
def _registers(device, warps):
    """
    The registers a thread may hold in a block of `warps` warps: as many as the card has for the block, and at most 255.
    Left to itself, ptxas answers a kernel that needs more than it can give by holding 40 and spilling everything else, for the occupancy.
    """
    import torch

    return min(255, torch.cuda.get_device_properties(device).regs_per_multiprocessor // (32 * warps))


def _spills(kernel, tile, *args, **kwargs):
    """How many registers `kernel` spills per thread at `tile`, compiled for `args` and loaded, or infinitely many if it does not compile."""
    import torch
    from triton.errors import TritonError

    shape, warps, stages = tile
    device = next(a.device for a in args if torch.is_tensor(a))
    try:
        compiled = kernel.warmup(*args, **kwargs, **_sizes(shape), num_warps=warps, num_stages=stages, maxnreg=_registers(device, warps), grid=(1,))
        compiled._init_handles()
    except TritonError:
        return math.inf
    return compiled.n_spills


def _widest_clean(kernel, tiles, *args, **kwargs):
    """
    The widest tile this kernel compiles for without spilling, and every narrower one.

    Spilling costs far more than a wider tile wins, and where the threshold falls depends on
    the operator and the card, so it is compiled and measured rather than guessed. Narrower
    tiles give each thread fewer elements and so cannot need more registers, which is why the
    search can stop at the first clean one.
    """
    ordered = sorted(tiles, key=_load, reverse=True)
    for i, tile in enumerate(ordered):
        if _spills(kernel, tile, *args, **kwargs) == 0:
            return ordered[i:]
    return ordered[-1:]


def _affordable(kernel, tiles, device, *args, **kwargs):
    """
    `tiles` less those whose spills would not fit: the driver reserves local memory for them on behalf of every thread the card can hold, and a card that cannot find it resets.
    So a tile that would need more than a sixteenth of the card that way is dropped, unless nothing needs less.
    """
    import torch

    card = torch.cuda.get_device_properties(device)
    threads = card.multi_processor_count * card.max_threads_per_multi_processor
    local = {tile: 4 * _spills(kernel, tile, *args, **kwargs) * threads for tile in tiles}
    return [tile for tile in tiles if local[tile] <= card.total_memory / 16] or [min(tiles, key=local.get)]


def _stripes(extents, tiles, device):
    """
    How many copies a gradient shared along the outermost axis is summed into: one per block along that axis, as far as the card runs them at once.
    The copies are allocated before the autotuner picks a tile, so there are as many as whichever of `tiles` needs the most, and the others leave some at zero.
    """
    import triton

    # Lists rather than generators, which torch.compile cannot trace into math.prod or max.
    return max([max(1, min(triton.cdiv(extents[0], shape[0]), _programs(device) // math.prod([triton.cdiv(e, t) for e, t in zip(extents[1:], shape[1:])]))) for shape, _, _ in tiles])


def _plane(shapes):
    """The shape the operands broadcast to, which is what the kernel writes."""
    import torch

    sized = [shape for shape in shapes if shape is not None]
    if not any(sized):
        raise Unsupported('nothing to tile over')
    try:
        return tuple(torch.broadcast_shapes(*sized))
    except RuntimeError as error:
        raise Unsupported(f'{sized} do not broadcast') from error


@functools.lru_cache(maxsize=4096)
def _layout(shapes):
    """
    The plane the operands broadcast to, the same plane in as few axes as they allow, and which of those axes each operand varies along.

    However many axes a multivector's coefficients carry they are contiguous, so two adjacent axes read as one wherever every operand varies along both or along neither.
    What is left tells the operands apart by the axes they vary along rather than by their rank: an input varies along all of them, a weight along the last, and the input to a fully connected product along all but the output features it is shared over.
    A plain number -- the ``math.sqrt(2)`` a layer divides by -- rides along by value and varies along nothing, as ``None``.
    Operands are right-aligned and broadcast against each other the way torch would.
    The call path asks this too, to tell a layout the generated code already covers from one that needs its own.
    That is twice per call, which is why it is cached: working it out costs more than a small kernel takes to run.
    """
    plane = _plane(shapes)
    aligned = [None if shape is None else (1,) * (len(plane) - len(shape)) + tuple(shape) for shape in shapes]
    extents, columns = [], []
    for k, extent in enumerate(plane):
        # Under torch.compile a size can be symbolic. Branching settles each test to a plain bool (bool() does not), where comparing columns of symbolic ones fails in sympy.
        column = tuple(True if shape is not None and shape[k] != 1 else False for shape in aligned)
        if extent == 1:
            continue
        if columns and columns[-1] == column:
            extents[-1] *= extent
        else:
            extents.append(extent)
            columns.append(column)
    if not extents:
        extents, columns = [1], [tuple(shape is not None for shape in aligned)]
    varies = [None if shape is None else tuple(column[i] for column in columns) for i, shape in enumerate(aligned)]
    return plane, tuple(extents), tuple(varies)


def _plan(bases, shapes):
    _, extents, varies = _layout(shapes)
    return [replace(base, varies=v, shape=shape) for base, shape, v in zip(bases, shapes, varies)], extents


def _var(value):
    """The name of the symbol a coefficient is, or ``'_'`` for one that is not a symbol, a structural zero say."""
    return value.name if isinstance(value, sympy.Symbol) else rp_var_name(value)


def _sympy_body(exprs):
    """
    :func:`_body` for sympy expressions.
    Temporaries start with an underscore, as the names the kernel gives its own things do, clear of the symbols of the expressions.
    """
    printer = TritonPrinter()
    pairs, outs = sympy.cse(exprs, symbols=sympy.numbered_symbols('_t'))
    try:
        lines, outs = [f'    {name} = {printer.doprint(e)}' for name, e in pairs], [printer.doprint(e) for e in outs]
    except PrintMethodNotImplementedError as error:
        raise Unsupported(str(error)) from error
    # A function that triton does not have is printed from the module python has it in.
    if foreign := set(printer.module_imports) - {'tl'}:
        raise Unsupported(f'functions from {foreign}')
    return lines, outs


def _body(exprs):
    """CSE'd assignment lines and one formatted expression per output."""
    exprs = list(exprs)
    if all(isinstance(e, sympy.Expr) for e in exprs):
        return _sympy_body([sympy.factor_terms(e.evalf()) for e in exprs])
    if not all(isinstance(e, RationalPolynomial) for e in exprs):
        raise Unsupported('not polynomial')
    divided = [e for e in exprs if e.denom != 1]
    if any(e.denom != divided[0].denom for e in divided):
        raise Unsupported('denominators differ')

    pairs, numer, denom = rational_cse(exprs, divided[0].denom if divided else None)
    lines = [f'    {name} = {poly_format(poly)}' for name, poly in pairs]
    quotient = '_d'
    while any(quotient == name for name, _ in pairs):
        quotient += '_'
    if denom is not None:
        lines.append(f'    {quotient} = {poly_format(denom)}')
    outs = [poly_format(p) if denom is None or e.denom == 1 else f'({poly_format(p)})/({quotient})' for e, p in zip(exprs, numer)]

    found = {root for e in exprs for root in e.roots}
    lines, outs = _expand_roots(lines, found), _expand_roots(outs, found)
    if any('**' in text for text in (*lines, *outs)):
        raise Unsupported('fractional power')  # an integer one expands to x*x
    return lines, outs


def _expand_roots(texts, roots):
    """Rewrite each ``x**0.5`` symbol as a tl.sqrt call, longest first so nested roots nest."""
    for root in sorted(roots, key=len, reverse=True):
        if not any(root in text for text in texts):
            continue
        base = root.base
        inner = (poly_format(base.numer.args) if base.denom == 1
                 else f'({poly_format(base.numer.args)})/({poly_format(base.denom.args)})')
        texts = [text.replace(root, f'tl.sqrt({inner})') for text in texts]
    return texts


def _tag(varies):
    return ''.join('1' if v else '0' for v in varies)


def _at(varies, i):
    """Where coefficient `i` of an operand varying along `varies` sits, past its pointer."""
    return f'{i} * _n_{_tag(varies)} + _o_{_tag(varies)}'


def _range(start, k, n):
    """The block of axis `k` from `start`, promised contiguous along the innermost axis so that its loads vectorise."""
    index = f'{start} + tl.arange(0, T{k})'
    return f'tl.max_contiguous(tl.multiple_of({index}, T{k}), T{k})' if k == n - 1 else index


def _blocks(n):
    """
    Where this program's block sits along each axis, the innermost varying fastest, leaving in `_pid` its block along the outermost -- which picks the stripe a shared gradient goes into.

    What the kernel names for itself starts with an underscore, clear of the temporaries the expressions name.
    """
    lines = ['    _pid = tl.program_id(0)', '    if WIDE:', *(f'        e{k} = e{k}.to(tl.int64)' for k in range(n))]
    for k in reversed(range(1, n)):
        lines += [f'    _i{k} = {_range(f"(_pid % tl.cdiv(e{k}, T{k})) * T{k}", k, n)}', f'    _pid = _pid // tl.cdiv(e{k}, T{k})']
    return lines + [f'    _i0 = {_range("_pid * T0", 0, n)}']


def _address(n, patterns):
    """
    Per axis its offsets broadcast to the tile and its mask, and per pattern in `patterns` the offsets, mask and blade stride of an operand varying along it.

    An operand is read at its own shape, so one shared along an axis is loaded once per block rather than once per element of it, and broadcasts in registers.
    """
    lines = []
    for k in range(n):
        lines += [f'    _x{k} = _i{k}' + (f'[{", ".join(":" if j == k else "None" for j in range(n))}]' if n > 1 else ''), f'    _m{k} = _x{k} < e{k}']
    for varies in patterns:
        along = [k for k, v in enumerate(varies) if v]
        tag = _tag(varies)
        lines += [f'    _o_{tag} = ' + (' + '.join(f'_x{k}' + ''.join(f' * e{j}' for j in along if j > k) for k in along) or '0'),
                  f'    _n_{tag} = ' + (' * '.join(f'e{k}' for k in along) or '1'),
                  f'    _m_{tag} = ' + (' & '.join(f'_m{k}' for k in along) or 'None')]
    return lines


def _loads(ops):
    return [f'    {var} = tl.load({op.name} + {_at(op.varies, i)}, mask=_m_{_tag(op.varies)})' for op in ops for i, var in op.live]


@functools.cache
def _programs(device):
    """Enough programs to fill the card several times over, and so as many copies of a shared gradient as are worth keeping apart."""
    import torch

    return 8 * torch.cuda.get_device_properties(device).multi_processor_count


def _params(plan):
    """Kernel parameters: a pointer per array argument, a value per scalar one."""
    return [var for op in plan for var in (op.vars if not op.array else [op.name])]


def _sizes(shape):
    return {f'T{k}': size for k, size in enumerate(shape)}


def _choose_tiles(namespace, funcname, plan, values, n_out, extents, span=None, tiles=None):
    """
    Autotune the forward over the tiles that do not spill, and the backward over the same sizes in every shape :func:`_fit` gives them.
    The backward has no size of its own to start from: whatever the forward can hold without spilling is where its own spills are still worth timing.
    Nothing is spill-free there for a large algebra, and a wide tile that spills a little beats a narrow one that does not, so the timing decides among whichever :func:`_affordable` lets it run.
    """
    import torch
    import triton

    reference = next(v for v in values if torch.is_tensor(v))
    call = [x for v, op in zip(values, plan) for x in ((v,) if op.array else v)]
    out = torch.empty((n_out, math.prod(extents)), device=reference.device, dtype=reference.dtype)
    saved = [out.new_empty(extents[0] * namespace['_KEPT'].value)] if namespace['_KEPT'].value else []
    scratch = [out.new_empty(extents[0] * namespace['_STAGED'].value)] if namespace['_STAGED'].value else []
    wide = math.prod(extents) * (span or max(n_out, *(op.slots for op in plan if op.array))) >= 2 ** 31
    sizes = [f'e{k}' for k in range(1, len(extents))]

    def autotune(kernel, tiles, key, **kwargs):
        configs = [triton.Config(_sizes(shape), num_warps=w, num_stages=s, maxnreg=_registers(reference.device, w)) for shape, w, s in tiles]
        return triton.autotune(configs=configs, key=key, **kwargs)(kernel)

    widths = _widths(extents)
    clean = _widest_clean(namespace[f'{funcname}_fwd'], tiles or _tiles(widths, CONFIGS), *call, out, *saved, *extents, wide)
    namespace[f'{funcname}_fwd'] = autotune(namespace[f'{funcname}_fwd'], clean, sizes)

    elements = list(dict.fromkeys((math.prod(shape), w, s) for shape, w, s in clean))
    candidates = _tiles(widths, elements, [1 << k for k in range(max(e for e, _, _ in elements).bit_length())])
    arrays = [(v, op) for v, op in zip(values, plan) if op.array]
    grads = {f'd{op.name}': v if all(op.varies) else v.to(torch.promote_types(v.dtype, torch.float32)) for v, op in arrays}
    flags = {f'G{op.name}': True for _, op in arrays}
    stripes = _stripes(extents, candidates, reference.device)
    bwd = [*call, *extents, *grads.values(), out, *saved, *scratch, stripes]
    kernel = namespace[f'{funcname}_bwd']
    # A kernel over given tiles, a whole layer's, times only those of them its backward holds without spilling, if any: a spilling one never won there.
    namespace['_BACKWARD'] = _affordable(kernel, _widest_clean(kernel, candidates, *bwd, WIDE=wide, **flags) if tiles else candidates, reference.device, *bwd, WIDE=wide, **flags)
    namespace[f'{funcname}_bwd'] = autotune(kernel, namespace['_BACKWARD'], sizes + list(flags), reset_to_zero=[f'd{op.name}' for _, op in arrays if not all(op.varies)])


def triton_lambdify(args, exprs, funcname, cse=True, output_mv_idx=None, values_asarray=None, shapes=None, keys=None, printer=None):
    """
    A differentiable callable over stacked coefficient tensors, backed by a Triton kernel.
    An operator with einops calls in it is one kernel over blocks of rows, which contracts and reduces the features of its rows in registers.

    :param shapes: ``{argument name: shape}``, from :func:`~kingdon.codegen.do_compile_symbolic`.
    :param keys: ``{argument name: keys}``, from the same.
    :param printer: what :func:`~kingdon.codegen.lambdify` prints sympy expressions with, where there is no kernel.
    """
    from kingdon.codegen import lambdify

    plain = lambdify(args, exprs, funcname, printer=printer, cse=cse, output_mv_idx=output_mv_idx, values_asarray=values_asarray, shapes=shapes, keys=keys)
    fused = any(isinstance(e, sympy.Basic) and e.has(ArrayBase) for e in exprs)
    try:
        if output_mv_idx is not None:
            raise Unsupported('writes into an argument')
        if fused:
            exprs = [sympy.sympify(e) for e in exprs]
            ranks = _feature_ranks(exprs)
        else:
            lines, outs = _body(exprs)
    except Unsupported:
        return plain

    import torch

    built = {}
    bases = []
    for name, vals in args.items():
        nested = any(isinstance(v, (list, tuple)) for v in vals)
        bases.append(Operand(name, tuple(_var(v) for v in (vals[0] if nested else vals)), nested))
    if fused:
        franks = tuple(max((ranks.get(sympy.Symbol(var), 99) for _, var in base.live), default=99) for base in bases)

    def datashape(value, base):
        return tuple(value.shape[base.lead:]) if torch.is_tensor(value) else None

    def signature(values):
        """
        Everything the generated code depends on: the axes each operand varies along, how many
        coefficients it carries, its dtype, and the widths its tiles are shaped to. Not the
        extents themselves -- the kernel takes those as arguments, so one build serves every size
        those tiles cover.

        The leading axes are still checked exactly, because a multivector torch would broadcast
        has to be turned away rather than read as though it had coefficients it does not have.
        """
        if len(values) != len(bases):
            return None
        shapes_in, dtypes = [], []
        for value, base in zip(values, bases):
            # A scalar has to be plain numbers: the kernel takes it by value and reports no
            # gradient, which only holds for a constant. A zero-dimensional tensor is an
            # operand like any other and arrives stacked, so it is not this case.
            if isinstance(value, (list, tuple)):
                if len(value) != base.slots or any(torch.is_tensor(v) for v in value):
                    return None
                shapes_in.append(None)
                dtypes.append(None)
            elif not torch.is_tensor(value) or value.is_cpu:
                return None
            elif tuple(value.shape[:base.lead]) != ((1, base.slots) if base.nested else (base.slots,)):
                return None
            else:
                shapes_in.append(datashape(value, base))
                dtypes.append(value.dtype)
        try:
            if fused:
                return tuple(zip(_fused_layout(tuple(shapes_in), franks)[1], dtypes))
            _, extents, varies = _layout(tuple(shapes_in))
        except Unsupported:
            return None
        return tuple(zip(varies, dtypes)), _widths(extents)

    def build(values):
        """
        Emit the kernel from the layout the operands actually arrive with.

        `shapes` cannot be trusted for this. An operator is cached on (type, keys), and the
        first call that creates it is often the symbolic one inside another operator's
        codegen, where every multivector is shapeless -- so whether an operator got a kernel
        would depend on the order codegen happened to reach it in.
        """
        datashapes = tuple(datashape(value, base) for value, base in zip(values, bases))
        dtype = functools.reduce(torch.promote_types, [v.dtype for v in values if torch.is_tensor(v)])
        if fused:
            return fused_build(values, datashapes, dtype)
        plan, extents = _plan(bases, datashapes)
        grad_lines, grads = _gradients(plan, exprs)
        return run(_source(funcname, plan, len(extents), lines, outs, grad_lines, grads, dtype), plan, values, len(outs), extents)

    def fused_build(values, datashapes, dtype):
        batch, tiles = _fused_layout(datashapes, franks)
        plan = [replace(base, varies=tile and (tile[0],), shape=shape) for base, shape, tile in zip(bases, datashapes, tiles)]
        leaves = {sympy.Symbol(var): tile or (False, ()) for op, tile in zip(plan, tiles) for _, var in op.live}
        # As torch's own matmul, on tensor cores: a float32 dot on the cores that do fused multiply-adds instead holds so much of both operands per thread that it spills. tf32x3 is as close to float32 as tensor cores get.
        precision = 'ieee' if dtype != torch.float32 or torch.version.hip else 'tf32x3' if torch.get_float32_matmul_precision() == 'highest' else 'tf32'
        printer = TritonPrinter(dict(leaves), precision)
        out_tile = (True, tuple(np.broadcast_shapes(*(printer.shape(e)[1] for e in exprs))))
        grad_printer = TritonPrinter({**leaves, **{sympy.Symbol(f'go{k}'): out_tile for k in range(len(exprs))}}, precision)
        gradients = _fused_gradients(plan, exprs, grad_printer)
        # A tile holds every feature of its rows, and of the whole layer, so a block has the fewest rows a tl.dot takes, 16, and one warp: triton lays the dots of a kernel with several
        # out along the rows, so another warp would hold the same rows again, and a warp of its own keeps every change of layout to shuffles within it.
        rows = [((16,), 1, 1)]
        # What the values of a segment may take of a thread's registers, see _split, as single precision ones: a quarter stays for what that does not count --
        # indices, masks and addresses, the fragments of a tl.dot, a tile changing layout.
        budget = _registers(next(v for v in values if torch.is_tensor(v)).device, 1) * 3 // 4 * 4 // dtype.itemsize
        span = max(len(exprs) * math.prod(out_tile[1]), *(op.slots * math.prod(tile[1]) for op, tile in zip(plan, tiles) if op.array))
        return run(_fused_source(funcname, plan, tiles, gradients, out_tile, printer, grad_printer, span, dtype, budget), plan, values, len(exprs), (math.prod(batch),), span, rows)

    def run(src, plan, values, n_out, extents, *tuning):
        filename = f'{funcname}#{next(_BUILDS)}'
        namespace = {'_layout': _layout, '_stripes': _stripes}
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        exec(compile(src, filename, 'exec'), namespace)
        _choose_tiles(namespace, funcname, plan, values, n_out, extents, *tuning)
        return namespace[funcname]

    def dispatch(*values):
        # Symbolic calls come through here too, during another operator's codegen, and carry
        # polynomials rather than tensors. Build from nothing and every argument looks shapeless,
        # so wait for real ones rather than remembering the failure.
        if not any(torch.is_tensor(v) and not v.is_cpu for v in values):
            return plain(*values)
        key = signature(values)
        if key is None:
            return plain(*values)
        if key not in built:
            try:
                built[key] = build(values)
            except Unsupported:
                built[key] = False
        kernel = built[key]
        return kernel(*values) if kernel else plain(*values)

    dispatch.__name__ = funcname
    return dispatch


def _gradients(plan, exprs):
    """d(sum_k go_k * out_k)/ds per input symbol, CSE'd together so they share work."""
    symbol = RationalPolynomial.fromname if isinstance(exprs[0], RationalPolynomial) else sympy.Symbol
    go = [symbol(f'go{k}') for k in range(len(exprs))]
    loss = go[0] * exprs[0]
    for g, e in zip(go[1:], exprs[1:]):
        loss = loss + g * e
    names = [var for op in plan if op.array for _, var in op.live]
    lines, formatted = _body([loss.diff(symbol(var)) for var in names])
    return lines, dict(zip(names, formatted))


def _feature_ranks(exprs):
    """
    For every symbol, how many of the trailing axes of its coefficients are features, the axes it has before those being rows: as many as the tile it enters has.
    An einsum pattern names them, a reduction or a reshape changes their count, and anything pointwise has as many as what it combines with.
    """
    ranks, known = {}, {}

    def enters(e, rank):
        if isinstance(e, sympy.Symbol):
            ranks[e] = max(ranks.get(e, 0), rank)
        elif known.get(e) is None:
            for a in e.args:
                enters(a, rank)

    def visit(e):
        if isinstance(e, Einsum):
            _, _, terms, right = _unbladed(e, [isinstance(op, Stack) for op in e.args[1:]])
            for op, term in zip(e.args[1:], terms):
                enters(op, len(term.replace('...', '')))
            known[e] = len(right.replace('...', ''))
        elif isinstance(e, (Reduce, Reshape)):
            a, k, sizes = e.args
            rank = known[a] if known.get(a) is not None else -int(min(sizes)) if isinstance(e, Reduce) else int(k) - 1
            enters(a, rank)
            known[e] = rank - len(sizes) if isinstance(e, Reduce) else len(sizes) - 1 if k > rank else rank - int(k) + len(sizes)
        elif counts := [known[a] for a in e.args if known.get(a) is not None]:
            known[e] = max(counts)
            for a in e.args:
                enters(a, known[e])
        return e

    visit = _bottom_up(visit)
    for e in exprs:
        visit(e)
    return ranks


@functools.lru_cache(maxsize=4096)
def _fused_layout(shapes, ranks):
    """
    The rows the operands share, and each operand's tile, or None for a plain number: its axes before the trailing `ranks` features are rows, if any of them is not one.
    An operand with rows has all of them, since one broadcast along some rows only would need a row axis of its own.
    """
    split = [None if shape is None else (shape[:len(shape) - min(len(shape), rank)], shape[len(shape) - min(len(shape), rank):]) for shape, rank in zip(shapes, ranks)]
    try:
        batch = np.broadcast_shapes(*(rows for rows, _ in filter(None, split)))
    except ValueError as error:
        raise Unsupported(f'{shapes} do not broadcast') from error
    tiles = [None if s is None else (math.prod(s[0]) > 1, s[1]) for s in split]
    if any(tile and tile[0] and (1,) * (len(batch) - len(s[0])) + s[0] != batch for s, tile in zip(split, tiles)):
        raise Unsupported(f'{shapes} broadcast along some rows only')
    return batch, tiles


def _reshaped(e, tile, features):
    """`e`, of `tile`, as a Reshape node to `features`, which differ from its own in unit axes only."""
    rows, own = tile
    return e if own == features else Reshape(e, len(own) + rows, sympy.Tuple(*(-1,) * rows, *features))


def _unbroadcast(g, tile, target):
    """The cotangent `g`, of `tile`, summed over the rows and features it has broadcast beyond those of the tile `target`, as those."""
    (rows, features), (rows_to, to) = tile, target
    aligned = (0,) * (len(features) - len(to)) + to
    axes = [-1 - len(features)] * (rows and not rows_to) + [k - len(features) for k, (n, m) in enumerate(zip(features, aligned)) if n != m]
    kept = tuple(n for k, n in enumerate(features) if k - len(features) not in axes)
    return _reshaped(Reduce(g, Str('sum'), sympy.Tuple(*axes)), (rows and rows_to, kept), to) if axes else g


def _adjoints(node, g, shape):
    """Each operand of an einops node, with the gradient it gets from the cotangent `g` of the node; over multivectors, operand and cotangent are Stacks of blades."""
    if isinstance(node, Einsum):
        terms, right = _terms(node)
        for m, op in enumerate(node.args[1:]):
            others = [(term, o) for k, (term, o) in enumerate(zip(terms, node.args[1:])) if k != m]
            yield op, Einsum(Str(f'{",".join([right, *(term for term, _ in others)])}->{terms[m]}'), g, *(o for _, o in others))
        return
    a, operation, axes = node.args
    rows, features = shape(a)
    if isinstance(node, Reshape):
        adjoint = lambda g: _reshaped(g, shape(g), features)
    elif operation.name in ('sum', 'mean'):
        kept = tuple(1 if k - len(features) in axes else n for k, n in enumerate(features))
        adjoint = lambda g: _reshaped(g, shape(g), kept) / (math.prod(features) // math.prod(kept) if operation.name == 'mean' else 1)
    else:
        raise Unsupported(f'the gradient of {operation.name}')
    yield a, Stack(*map(adjoint, g.args)) if isinstance(g, Stack) else adjoint(g)


def _scattered(operand, adjoint):
    """
    The gradient `adjoint` of `operand` per element of it. A Stack gathers its elements into the blades of a multivector, a weight per grade into every blade of that grade say,
    so an element's gradient is the sum over the blades it went to: of an einsum, the same einsum contracting its blade axis there, which the kernel does as one chain of tl.dot.
    """
    if not isinstance(operand, Stack):
        yield operand, adjoint
        return
    for element in dict.fromkeys(operand.args):
        at = [i for i, a in enumerate(operand.args) if a == element]
        if len(at) == 1:
            yield element, Blade(adjoint, at[0])
        elif isinstance(adjoint, Einsum):
            terms, right = _terms(adjoint)
            yield element, Einsum(Str(f'{",".join(terms)}->{right[1:]}'), *(Stack(*(op.args[i] for i in at)) if isinstance(op, Stack) else op for op in adjoint.args[1:]))
        else:
            yield element, sympy.Add(*(adjoint.args[i] for i in at))


def _fused_gradients(plan, exprs, printer):
    """
    `exprs` as a program of named values, the forward, and the gradient of ``sum_k go_k * exprs[k]`` by the symbol of every array operand, the backward:
    the expressions of the outputs, the values they name, those values the backward may load rather than compute, and (expression, symbol) roots in the order a kernel computes them with the values they name.
    The forward is sympy's cse of `exprs`, and every value it names that has rows, and is not :func:`_cheap`, is cut out as a symbol, as is every einops node, a node over multivectors a symbol per blade.
    Reverse mode over the cuts, each once every cut using it is gone through, with sympy's diff within each.
    A value's gradient joins the terms ``gradient * value``, from which its operands gather theirs by diff in their turn: what waits is the gradient of the value, rather than a sum for each of its operands.
    A node's gradient goes back through it transposed, which sympy's diff does not do: its adjoints are added into its operands' gradients at once, each sum a root of its own, since a node feeds
    a whole multivector of gradients that would otherwise wait. A leaf's gradient is a root once no cut left to go through has it in its value.
    """
    defs, program, blades, cuts, roots, grads = {}, {}, {}, [], [], {}

    def define(name, value):
        s = sympy.Symbol(name)
        defs[s], printer.shapes[s], printer.counts[s] = value, printer.shape(value), printer.count(value)
        return s

    def rule(e):
        if e in program:
            return program[e]
        if isinstance(e, (Einsum, Reduce, Reshape)):
            y = define(f'_y{len(cuts)}', e)
            if printer.counts[y]:
                blades[e] = y = tuple(define(f'{y}_{i}', Blade(y, i)) for i in range(printer.counts[y]))
            cuts.append((e, y))
            return e if e in blades else y
        if isinstance(e, Blade):
            a, i = e.args[0], int(e.args[1])
            return a.args[i] if isinstance(a, Stack) else blades[a][i]
        return e

    visit = _bottom_up(rule)
    # Products named by a cut, as powers of their factors: sympy flattens a product into the products it takes part in, gate * y * w for H * w with H = gate * y say,
    # so a value would take H's factors again, rather than H.
    products = []

    def factored(e):
        """`e` with each product holding a named one, as many times as it does, as that name times the rest."""
        if not isinstance(e, sympy.Mul):
            return e
        powers = e.as_powers_dict()
        for s, named in products:
            if (k := min(powers.get(b, 0) // n for b, n in named.items())) > 0:
                e = e / sympy.Mul(*(b ** (n * k) for b, n in named.items())) * s ** k
                powers = e.as_powers_dict()
        return e

    pairs, outs = sympy.cse(exprs, symbols=sympy.numbered_symbols('_f'), order='none')
    for t, e in pairs:
        program[t] = v = _bottom_up(factored)(visit(e))
        if not isinstance(v, (sympy.Symbol, ArrayBase)) and printer.shape(v)[0] and not _cheap(v, defs):
            program[t] = define(str(t), v)
            cuts.append((v, program[t]))
            named = v.as_powers_dict() if isinstance(v, sympy.Mul) else {}
            if len(named) > 1 and all(not b.is_number and n.is_Integer and n > 0 for b, n in named.items()):
                products.insert(0, (program[t], named))
    outputs, forward = [_bottom_up(factored)(visit(e)) for e in outs], dict(defs)
    owner = {s: k for k, (_, y) in enumerate(cuts) for s in (y if isinstance(y, tuple) else [y])}
    operands = [{owner[s] for s in value.free_symbols if s in owner} for value, _ in cuts]
    users = [sum(k in ks for ks in operands) for k in range(len(cuts))]
    # How many cuts left to go through have a leaf in their value: none, and its gradient is complete.
    pending = {sympy.Symbol(var): sum(sympy.Symbol(var) in value.free_symbols for value, _ in cuts) for op in plan if op.array for _, var in op.live}

    def push(weight, value, cut=None):
        """
        Add ``weight * d value / d s``, summed to the tile of s, to the gradient of every cut and leaf s in `value`, the value of `cut`: at once, but for a leaf that cut completes, whose store adds it.
        A leaf without rows, a weight, gets every term as a root of its own, which the kernel adds into its gradient right away rather than hold a sum.
        """
        for s in _symbols(value):
            if (s in owner or s in pending) and (d := value.diff(s)) != 0:
                g = weight * d
                g = _unbroadcast(g, printer.shape(g), printer.shapes[s])
                if s in pending and not printer.shapes[s][0]:
                    roots.append((g, s))
                    continue
                grads[s] = define(f'_d{len(defs)}', grads[s] + g if s in grads else g)
                if pending.get(s, 2) > (cut is not None):
                    roots.append((grads[s], None))

    # The terms ``gradient * value`` of the values gone through: each value's gradient waits there until its operands gather theirs, by their diff.
    loss = []
    free = functools.cache(lambda term: term.free_symbols)

    def gathered(s):
        """The gradient by `s`: what the adjoints of nodes pushed, and the diff of the terms that have `s`, summed to its tile."""
        g = grads.get(s, sympy.S.Zero) + sympy.Add(*(term.diff(s) for term in loss if s in free(term)))
        return _unbroadcast(g, printer.shape(g), printer.shapes[s])

    def complete():
        for leaf in [leaf for leaf, n in pending.items() if n == 0]:
            if (g := gathered(leaf)) != 0 or printer.shapes[leaf][0]:
                roots.append((g, leaf))
            del pending[leaf]

    loss += [sympy.Symbol(f'go{k}') * e for k, e in enumerate(outputs)]
    complete()
    # A cut is gone through once every cut using it is, and an einsum first, since it turns a multivector of gradients into sums already held.
    ready = [k for k in range(len(cuts)) if not users[k]]
    while ready:
        ready.remove(k := max(ready, key=lambda k: (isinstance(cuts[k][0], Einsum), k)))
        value, y = cuts[k]
        g = [gathered(s) for s in (y if isinstance(y, tuple) else [y])]
        if any(g):
            g = [define(f'_d{len(defs)}', d) for d in g]
            roots.extend((d, None) for d in g)
            if isinstance(value, ArrayBase):
                for operand, adjoint in _adjoints(value, Stack(*g) if isinstance(y, tuple) else g[0], printer.shape):
                    for element, gradient in _scattered(operand, adjoint):
                        push(define(f'_g{len(defs)}', gradient), element, k)
            else:
                loss.append(g[0] * value)
        for leaf in value.free_symbols & pending.keys():
            pending[leaf] -= 1
        complete()
        for j in operands[k]:
            users[j] -= 1
            if not users[j]:
                ready.append(j)
    return outputs, forward, [s for s in owner if printer.shapes[s][0] and not _simple(forward[s])], roots, defs


def _simple(v):
    """Whether `v` takes a few multiplications and additions and nothing else: no function, no root, no einops."""
    return sympy.count_ops(v) <= 4 and not v.atoms(sympy.Function) and all(p.exp.is_Integer for p in v.atoms(sympy.Pow))


def _cheap(v, values):
    """
    Whether `v` is best computed where it is used, and differentiated there: :func:`_simple`, of at most one of `values` that is not.
    Its operands are then held anyway, and its gradient goes to what holds them.
    A value of two that are not, a gate times what it gates say, is better held itself, as would be both, and its gradient kept apart from theirs.
    """
    return _simple(v) and sum(s in values and not _simple(values[s]) for s in v.free_symbols) <= 1


def _symbols(e):
    """The symbols of `e`, in the order they appear."""
    return list(dict.fromkeys(s for s in sympy.preorder_traversal(e) if isinstance(s, sympy.Symbol)))


def _topological(values):
    """The symbols of `values`, each after the values it refers to."""
    order, seen = [], set()
    for root in values:
        todo = [(root, False)]
        while todo:
            s, done = todo.pop()
            if done:
                order.append(s)
            elif s in values and s not in seen:
                seen.add(s)
                todo += [(s, True), *((d, False) for d in _symbols(values[s]))]
    return order


def _emit(roots, defs, printer, load, seeds=(), stage=None, budget=math.inf, pinned=()):
    """
    The lines computing `roots`, (expression, sink) pairs, in order, each value handed to its sink, which returns the lines that store it.
    Every value is computed, and every leaf loaded, right before its first use.
    Each root gets its leaves itself, which the compiler merges with what its segment got already.
    `defs` names values the roots, and each other, refer to by symbol; those depending on the `seeds`, the incoming cotangents, are sums the backward holds.
    Where what the kernel holds would take more than `budget` registers a thread, it goes in segments, see :func:`_split`: what one takes from an earlier one is stored at the place `stage` gives,
    one value after another, see :func:`_places`, but a `pinned` one, which keeps its own.
    """
    # Einops nodes make a DAG whose shared parts are large, and ordering the arguments of a sum or product counts the nodes of each as a tree, so cse leaves them in their order.
    pairs, reduced = sympy.cse([*defs.values(), *(e for e, _ in roots)], symbols=sympy.numbered_symbols('_t'), order='none')
    # A multivector is never a value of its own: a Stack only names the tiles it holds, and a node over multivectors is computed blade by blade where each blade is used,
    # since the operands of all its blades at once, a tl.dot's fragments say, would take more registers than anything else in the kernel.
    tuples = {}
    for s, v in [*pairs, *zip(defs, reduced)]:
        if isinstance(v := v.xreplace(tuples), Stack) or isinstance(v, (Einsum, Reduce, Reshape)) and printer.count(v):
            tuples[s] = v

    @_bottom_up
    def split(e):
        """`e` with every blade of a multivector node that blade's own node, which names only what it takes, and a sum over the blades of multivectors the sum of those."""
        if isinstance(e, Blade) and isinstance(a := e.args[0], (Stack, Einsum, Reduce, Reshape)):
            return split(a.args[int(e.args[1])] if isinstance(a, Stack) else printer._blade(a, int(e.args[1])))
        if isinstance(e, Einsum) and (counts := [printer.count(op) for op in e.args[1:]]) and _unbladed(e, counts)[0] is not None and not printer.count(e):
            return sympy.Add(*(split(printer._blade(e, i)) for i in range(next(filter(None, counts)))))
        return e

    # A sum of more than a few multiplications is a value of its own, wherever it is, so that it goes a term at a time, see computed, with places between them a segment may start at.
    sums = {}

    @_bottom_up
    def lift(e):
        return sums.setdefault(e, sympy.Symbol(f'_u{len(sums)}')) if isinstance(e, sympy.Add) and not _simple(e) else e

    values = {s: lift(split(v.xreplace(tuples))) for s, v in [*pairs, *zip(defs, reduced)] if s not in tuples}
    exprs = [lift(split(e.xreplace(tuples))) for e in reduced[len(defs):]]
    values.update({u: e for e, u in sums.items()})
    # A value that is another's name is that value.
    aliases = {s: v for s, v in values.items() if isinstance(v, sympy.Symbol)}
    for s in aliases:
        while aliases[s] in aliases:
            aliases[s] = aliases[aliases[s]]
    values = {s: v.xreplace(aliases) for s, v in values.items() if s not in aliases}
    exprs = [e.xreplace(aliases) for e in exprs]
    backward = set(seeds)
    for s in _topological(values):
        if backward.intersection(_symbols(values[s])):
            backward.add(s)
    # A cheap value of the forward is computed by each root, and each segment, that takes it, from values held anyway, rather than stored for a later segment;
    # one of the backward is held, since computing it anew would hold what it adds up, and ptxas schedules those worse.
    cheap = {s for s, v in values.items() if s not in backward and _cheap(v, values)}
    for s in _topological(values):
        printer.shapes[s] = printer.shape(values[s])

    def needs(e):
        """What `e` needs, in the order to get it: values first, then the leaves, loaded right before the line that uses them."""
        return sorted(reversed(_symbols(e)), key=lambda s: s in values)

    def computed(s):
        """
        What to get, and in which order, to compute `s`: (symbol, None) to get a symbol, (s, (expression, last)) to compute s as that, and (s, ()) where a segment may start amid it.
        A sum the kernel holds goes a term at a time, each after what it takes, so that a root holds one term's operands and the sum so far.
        """
        v = values[s]
        terms = sorted(v.args, key=lambda t: t.is_number) if isinstance(v, sympy.Add) and s not in cheap else (v,)
        return [item for k, t in reversed(list(enumerate(terms))) for item in [(s, (s + t if k else t, k == len(terms) - 1)), *((d, None) for d in needs(t)), *[(s, ())] * bool(k)]]

    def run(starts=(), staged=()):
        """
        The lines of the kernel, and per symbol the lines that take it, the lines a segment may start at, the leaves, the held values, and the lines that contract, see take.
        A segment starts, after a barrier, at each of the places `starts` counts: at a root, or amid a sum, which it then takes from memory as far as it got.
        A `staged` value is stored where computed, and loaded again where a later segment uses it, at the place `stage` gives the value it maps to.
        """
        lines, taken, points, leaves, born, dots = [], {}, [], set(), {}, []
        segment = 0

        def take(*symbols, e=sympy.S.Zero):
            """The next line as one that takes `symbols` and those of `e`, and holds the fragments of its contractions."""
            for s in [*symbols, *_symbols(e)]:
                taken.setdefault(s, []).append(len(lines))
            # A tl.dot holds its operands once more, in the layout it takes them in, and tf32x3 splits each in two.
            if fragments := sum(_registers_of(printer.shape(op)) for node in e.atoms(Einsum) for op in node.args[1:]):
                dots.append((len(lines), fragments * (1 + (printer.precision == 'tf32x3'))))

        def point(local, partial=None):
            nonlocal segment
            points.append(len(lines))
            if len(points) - 1 in starts:
                lines.extend([_store_line(*stage(staged[partial]), partial), '    tl.debug_barrier()', _load_line(*stage(staged[partial]), partial)] if partial else ['    tl.debug_barrier()'])
                local.clear()
                segment += 1

        for (_, sink), e in zip(roots, exprs):
            local = set()
            point(local)
            todo = [(s, None) for s in needs(e)]
            while todo:
                s, line = todo.pop()
                if line == ():
                    point(local, s)
                    continue
                if line:
                    take(s, e=line[0])
                    # The first term of a sum may broadcast against the others, and the sum so far be stored where a segment starts amid it.
                    text = printer.doprint(line[0]) if printer.shape(line[0]) == printer.shapes[s] else f'tl.broadcast_to({printer.doprint(line[0])}, {_padded(printer.shapes[s])})'
                    lines.append(f'    {s} = {text}')
                    if line[1] and s in born:
                        born[s] = segment
                        lines += [_store_line(*stage(staged[s]), s)] if s in staged else []
                    continue
                if s in local:
                    continue
                local.add(s)
                if s in values and s not in cheap and s not in born:
                    born[s] = segment
                    todo += computed(s)
                elif s not in values or s not in cheap and born[s] < segment:
                    take(s)
                    leaves.update({s} - values.keys())
                    lines += [_load_line(*stage(staged[s]), s)] if s in values else load(s)
                elif s in cheap:
                    todo += computed(s)
            take(e=e)
            # A value staged for a later segment may be what the root stores anyway, and in the same place.
            lines += [line for line in sink(printer.doprint(e)) if [line] != lines[-1:]]
        return lines, taken, [*points, len(lines)], leaves, born, dots

    try:
        lines, taken, points, leaves, born, dots = run()
        if budget < math.inf:
            starts = _split(taken, {s: _registers_of(printer.shapes[s]) for s in taken}, points, budget, leaves, dots)
            cuts = sorted(points[k] for k in starts)
            staged = [s for s in born if bisect.bisect(cuts, taken[s][0]) < bisect.bisect(cuts, taken[s][-1])]
            lines, *_ = run(starts, _places(staged, {s: (taken[s][0], taken[s][-1]) for s in staged}, printer.shapes, pinned))
    except PrintMethodNotImplementedError as error:
        raise Unsupported(str(error)) from error
    # A function that triton does not have is printed from the module python has it in.
    if foreign := set(printer.module_imports) - {'tl'}:
        raise Unsupported(f'functions from {foreign}')
    return lines


def _places(staged, spans, tiles, pinned=()):
    """
    Per staged value the one whose place in the stage it takes, given per value the lines from where it is first stored to where it is last loaded, and its tile:
    a place freed by a value gone, of the same tile, or its own, as a `pinned` value always has. So what a block stages is what it holds at once, not all it ever held,
    small enough for the caches rather than sent on to memory.
    """
    places, occupants = {}, {}
    for s in sorted(staged, key=lambda s: spans[s][0]):
        gone = next((r for r in occupants.get(tiles[s], []) if spans[r][1] < spans[s][0]), None) if s not in pinned else None
        places[s] = s if gone is None else places[gone]
        if gone is not None:
            occupants[tiles[s]].remove(gone)
        if s not in pinned:
            occupants.setdefault(tiles[s], []).append(s)
    return places


def _registers_of(tile):
    """What a thread holds of a value of `tile`: a warp's 16 rows of it, the rows of the tile a tl.dot gives a warp, over its 32 lanes."""
    rows, features = tile
    return max(1, (16 if rows else 1) * math.prod(map(_pad, features)) // 32)


def _split(taken, registers, points, budget, leaves, dots=()):
    """
    Which of the `points`, the lines a segment of the kernel may start at, and the end, it does start at, given per symbol the lines that take it, and per line that contracts
    the registers its fragments take, as (line, registers) in order.
    A segment holds a symbol to the last of its lines that takes it -- the compiler merges a load, or a value, it has already -- and from the first, or, for what it loads, a leaf
    or a value of an earlier segment, from its start, since ptxas issues the loads of a stretch without barriers first. Each segment runs as far as that takes at most `budget`
    registers a thread, and the barrier before the next keeps the compiler from merging across: what a segment takes from an earlier one goes through memory, as whole tiles,
    where ptxas would spill a register at a time.
    """
    def peak(a, b):
        pressure = np.zeros(b - a + 1)
        for s, lines in taken.items():
            if (i := bisect.bisect_left(lines, a)) < (j := bisect.bisect_left(lines, b)):
                pressure[0 if i or s in leaves else lines[i] - a] += registers[s]
                pressure[lines[j - 1] - a + 1] -= registers[s]
        for line, fragments in dots[bisect.bisect_left(dots, (a,)):bisect.bisect_left(dots, (b,))]:
            pressure[line - a] += fragments
            pressure[line - a + 1] -= fragments
        return pressure.cumsum().max()

    starts, start = set(), 0
    # A segment only takes more as it runs further, so its end is a bisection away.
    while (start := max(start + 1, bisect.bisect(range(len(points)), budget, start + 1, key=lambda end: peak(points[start], points[end])) - 1)) < len(points) - 1:
        starts.add(start)
    return starts


def _store_line(pointers, mask, s):
    return f'    tl.store({pointers}, {s}{f", mask={mask}" if mask else ""})'


def _load_line(pointers, mask, s):
    return f'    {s} = tl.load({pointers}{f", mask={mask}, other=0.0" if mask else ""})'


def _source(funcname, plan, n, lines, outs, grad_lines, grads, dtype):
    arrays = [op for op in plan if op.array]
    full, tile = (True,) * n, f'[{", ".join(f"T{k}" for k in range(n))}]'
    stores = []
    for op in arrays:
        stores.append(f'    if G{op.name}:')
        for i, var in op.live:
            at, mask = f'd{op.name} + {_at(op.varies, i)}', f'_m_{_tag(op.varies)}'
            if all(op.varies):
                stores.append(f'        tl.store({at}, {grads[var]}, mask={mask})')
                continue
            # A shared operand's gradient is summed by the adds themselves: every element of the tile adds into the coefficient it read. Summing the tile first would cost a
            # trip through shared memory and a barrier per coefficient, since the axes it is shared along are spread over warps.
            if not op.varies[0]:
                at += f' + (_pid % STRIPES) * {op.slots} * _n_{_tag(op.varies)}'
            stores.append(f'        tl.atomic_add(tl.broadcast_to({at}, {tile}), {grads[var]}, mask=_m_{_tag(full)}, sem="relaxed")')
    index = [*_blocks(n), *_address(n, dict.fromkeys([full, *(op.varies for op in arrays)]))]
    return _module(funcname, plan, n, [*index, *_loads(arrays), *lines, *(f'    tl.store(out + {_at(full, k)}, {e}, mask=_m_{_tag(full)})' for k, e in enumerate(outs))],
                   [*index, *_loads(arrays), *(f'    go{k} = tl.load(gout + {_at(full, k)}, mask=_m_{_tag(full)})' for k in range(len(outs))), *grad_lines, *stores], len(outs),
                   f'data, extents, _ = _layout(({", ".join(f"{op.name}.shape[{op.lead}:]" for op in arrays)},))', max(len(outs), *(op.slots for op in arrays)), dtype)


def _fused_source(funcname, plan, tiles, gradients, out_tile, printer, grad_printer, span, dtype, budget=math.inf):
    """
    :func:`_source` for a kernel over blocks of rows, which holds every coefficient as a tile of all its features, see :func:`_fused_layout`.
    The forward and the backward are those of :func:`_fused_gradients`, every value and load of either emitted where :func:`_emit` needs it.
    Every value of the forward that has rows, and takes more than :func:`_simple` arithmetic, the backward loads rather than computes, from where the forward stores it:
    to compute them the backward would hold them, dots and gates of the whole layer, besides its own sums, see :func:`_registers_of`.
    """
    outputs, forward, saved, roots, defs = gradients
    arrays = [(op, tile) for op, tile in zip(plan, tiles) if op.array]
    where = {sympy.Symbol(var): (op, i, tile) for op, tile in arrays for i, var in op.live}
    slots = {}

    def at(name, i, tile, stride=None):
        """The pointers and the mask of coefficient i of an operand of `tile`, whose rows are flattened, and follow each other `stride` apart if given."""
        rows, features = tile
        axes = _axes(tile)
        offsets = [f'{_expand(index, k, len(axes))} * {stride if stride and k < rows else math.prod(features[k - rows + 1:])}' for k, (index, _) in enumerate(axes)]
        return ' + '.join([name, f'{i} * {"e0 * " * rows}{math.prod(features)}', *offsets]), ' & '.join(_expand(mask, k, len(axes)) for k, (_, mask) in enumerate(axes) if mask)

    def keep(s, slots=slots, name='saved', shapes=grad_printer.shapes):
        """
        Where the forward stores the value `s` for the backward, or a segment a value it stages for a later one: after the values given a place before it in each row of `name`.
        A row holds all of them, `stride` apart, so that each is the same pointers plus a constant, which the compiler folds into the load rather than hold an address per value.
        """
        if s not in slots:
            slots[s] = (_size(slots), shapes[s])
        offset, tile = slots[s]
        return at(f'{name} + {offset}', 0, tile, '_KEPT' if name == 'saved' else '_STAGED')

    held = {}
    hold = functools.partial(keep, slots=held, name='scratch')
    # Each value a multiple of 4 coefficients a row, so that every slot starts 16 bytes aligned, which a vector load needs.
    _size = lambda slots: sum(-(-math.prod(tile[1]) // 4) * 4 for _, tile in slots.values())

    def load(s):
        """A leaf's load, or nothing for a number the kernel takes by value."""
        if s in saved:
            pointers, mask = keep(s)
        elif s.name.startswith('go'):
            pointers, mask = at('gout', int(s.name[2:]), out_tile)
        elif s in where:
            op, i, tile = where[s]
            pointers, mask = at(op.name, i, tile)
        else:
            return []
        return [_load_line(pointers, mask, s)]

    def store(pointers, mask):
        return lambda value: [_store_line(pointers, mask, value)]

    def gradient(s):
        """Where the gradient by `s` goes: stored, for an operand with rows, else added into this block's stripe."""
        op, i, tile = where[s]
        pointers, mask = at(f'd{op.name}', i, tile)
        mask = f', mask={mask}' if mask else ''
        if tile[0]:
            return lambda value: [f'    if G{op.name}:', f'        tl.store({pointers}, {value}{mask})']
        return lambda value: [f'    if G{op.name}:', f'        tl.atomic_add({pointers} + (_pid % STRIPES) * {op.slots * math.prod(tile[1])}, {value}{mask}, sem="relaxed")']

    index = ['    _pid = tl.program_id(0)', '    _x0 = _pid * T0 + tl.arange(0, T0)', '    if WIDE:', '        _x0 = _x0.to(tl.int64)', '    _m0 = _x0 < e0']
    saved = set(saved)
    printer.zeroed, grad_printer.zeroed = set(where), {*where, *saved, *(sympy.Symbol(f'go{k}') for k in range(len(outputs)))}
    bwd = _emit([(g, gradient(s) if s else lambda _: []) for g, s in roots], {s: v for s, v in defs.items() if s not in saved}, grad_printer, load, [sympy.Symbol(f'go{k}') for k in range(len(outputs))], hold, budget)
    fwd = _emit([*((s, store(*keep(s))) for s in forward if s in slots), *((e, store(*at('out', k, out_tile))) for k, e in enumerate(outputs))], forward, printer, load,
                stage=functools.partial(keep, shapes=printer.shapes), budget=budget, pinned=set(slots))
    batch = ', '.join(f'{op.name}.shape[{op.lead}:{op.name}.ndim - {len(tile[1])}]' for op, tile in arrays)
    kept, scratch = _size(slots), _size(held)
    return _module(funcname, plan, 1, [*index, *fwd], [*index, *bwd], len(outputs), f'batch = torch.broadcast_shapes({batch}); data, extents = (*batch, *{out_tile[1]}), (math.prod(batch),)',
                   max(span, kept, scratch), dtype, kept, scratch)


def _module(funcname, plan, n, fwd, bwd, n_out, layout, span, dtype, kept=0, staged=0):
    """
    The forward and backward kernels, from the bodies `fwd` and `bwd`, and the autograd function that launches them:
    `layout` sets the output's `data` shape and the grid's `extents`, the forward stores `kept` coefficients a row in `saved`, for the backward and for its own later segments,
    and the backward `staged` a row in `scratch`, for its later segments.
    """
    arrays = [op for op in plan if op.array]
    saved, scratch = ', saved' if kept else '', ', scratch' if staged else ''
    names = [op.name for op in arrays]
    flags = [f'G{name}' for name in names]
    # An operand shared along the outermost axis -- a weight, shared over the batch -- sums its gradient into one of several copies, picked by block and added up by the
    # launcher, so that the blocks along that axis do not all contend for the same one.
    shared = any(not op.varies[0] for op in arrays)
    sizes = ", ".join(f"e{k}" for k in range(n))
    # Every constexpr follows every runtime argument: inductor launches a user kernel without its constexprs, yet finds the gradients to zero between timed configs by
    # their position in the whole signature. WIDE precedes the tile so that :func:`_widest_clean` can pass it positionally.
    constexprs = f'WIDE: tl.constexpr, {", ".join(f"T{k}: tl.constexpr" for k in range(n))}'

    fwd_args = f'{", ".join(_params(plan))}, out{saved}'
    bwd_args = lambda sizes: f'{", ".join(_params(plan))}, {sizes}, {", ".join("d" + name for name in names)}, gout{saved}{scratch}'
    kernels = ['@triton.jit', f'def {funcname}_fwd({fwd_args}, {sizes}, {constexprs}):', *fwd, '', '',
               '@triton.jit', f'def {funcname}_bwd({bwd_args(sizes)}, STRIPES, {constexprs}, {", ".join(f"{f}: tl.constexpr" for f in flags)}):', *bwd, '', '']

    buffers = []
    for op, flag in zip(arrays, flags):
        # A gradient gathered from several blocks starts at zero and accumulates in at least single precision.
        make = (f'torch.empty_like({op.name})' if all(op.varies) else
                f'torch.zeros(({"" if op.varies[0] else "stripes, "}*{op.name}.shape,), device={op.name}.device, dtype=torch.promote_types({op.name}.dtype, torch.float32))')
        buffers.append(f'd{op.name} = {make} if {flag} else {op.name}.new_empty(0)')
    scalars = [var for op in plan if not op.array for var in op.vars]
    returns = ', '.join('None' if not op.array else f'(d{op.name}{"" if op.varies[0] else ".sum(0)"}.to({op.name}.dtype) if G{op.name} else None)' for op in plan)

    launcher = f"""
_KEPT, _STAGED = tl.constexpr({kept}), tl.constexpr({staged})


def _grid(meta):
    return ({" * ".join(f'triton.cdiv(meta["e{k}"], meta["T{k}"])' for k in range(n))},)


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {", ".join(op.name for op in plan)}):
        {"; ".join(f"{name} = {name}.contiguous()" for name in names)}
        {"; ".join(f"{', '.join(op.vars)}, = {op.name}" for op in plan if not op.array) or "pass"}
        {layout}
        out = torch.empty(({n_out}, *data), device={names[0]}.device, dtype={dtype})
        {f"saved = out.new_empty(extents[0] * {kept})" if kept else "pass"}
        ctx.save_for_backward({", ".join(names)}{saved})
        ctx.scalars, ctx.extents = ({", ".join(scalars)}{"," if scalars else ""}), extents
        {funcname}_fwd[_grid]({fwd_args}, *extents, WIDE=math.prod(extents) * {span} >= 2 ** 31)
        return out

    @staticmethod
    def backward(ctx, gout):
        {", ".join(names)}{saved}, = ctx.saved_tensors
        ({", ".join(scalars)}{"," if scalars else ""}) = ctx.scalars
        {", ".join(flags)}, = {", ".join(f"ctx.needs_input_grad[{i}]" for i, op in enumerate(plan) if op.array)},
        gout, extents = gout.contiguous(), ctx.extents
        stripes = {"_stripes(extents, _BACKWARD, gout.device)" if shared else "1"}
        {"; ".join(buffers)}
        {f"scratch = gout.new_empty(extents[0] * {staged})" if staged else "pass"}
        {funcname}_bwd[_grid]({bwd_args('*extents')}, stripes, WIDE=math.prod(extents) * {span} >= 2 ** 31, {', '.join(f'{f}={f}' for f in flags)})
        return {returns}


def {funcname}(*values):
    return _Fn.apply(*values)
"""
    return 'import math\nimport torch\nimport triton\nimport triton.language as tl\n\n\n' + '\n'.join(kernels) + launcher
