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

import functools
import itertools
import linecache
import math
from dataclasses import dataclass, replace

import sympy
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import PythonCodePrinter

from kingdon.polynomial import RationalPolynomial, poly_format, rational_cse, rp_var_name

#: Tiles to consider, as (elements, warps, stages). How many registers a tile needs is
#: not worth predicting -- measured against a count of the live coefficients the ratio ranged
#: from 0.92 to 2.09 -- so :func:`_widest_clean` compiles them and reads the spills instead.
#: :func:`_fit` shapes each of these to the plane before any of that.
CONFIGS = [(64, 8, 1), (64, 4, 1), (128, 4, 1), (512, 16, 1), (1024, 16, 2)]

_BUILDS = itertools.count()


class Unsupported(Exception):
    """Raised when an expression cannot be emitted as a kernel, so the caller falls back."""


class TritonPrinter(PythonCodePrinter):
    """Sympy expressions as triton code: numbers as floats, the functions triton has as its own, and powers as the products and roots it can take."""

    def __init__(self):
        super().__init__({'user_functions': {name: f'tl.{name}' for name in ('erf', 'exp', 'log', 'sin', 'cos')}})

    def _print(self, expr, **kwargs):
        if isinstance(expr, sympy.Basic) and expr.is_number and not expr.is_Integer:
            return repr(float(expr))
        return super()._print(expr, **kwargs)

    def _print_Pow(self, expr, rational=False):
        base, power = self.parenthesize(expr.base, PRECEDENCE['Pow']), expr.exp
        if power in (sympy.S.Half, -sympy.S.Half):
            return f'tl.{"sqrt" if power > 0 else "rsqrt"}({self._print(expr.base)})'
        if not power.is_Integer:
            raise Unsupported(f'power {power}')
        product = '*'.join([base] * abs(int(power)))
        return product if power > 0 else f'1/({product})'


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


def _spills(kernel, tile, *args, **kwargs):
    """How many registers `kernel` spills per thread at `tile`, compiled for `args` and loaded, or infinitely many if it does not compile."""
    from triton.errors import TritonError

    shape, warps, stages = tile
    try:
        compiled = kernel.warmup(*args, **kwargs, **_sizes(shape), num_warps=warps, num_stages=stages, grid=(1,))
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
    """:func:`_body` for sympy expressions."""
    exprs = [sympy.factor_terms(sympy.sympify(e).evalf()) for e in exprs]
    printer = TritonPrinter()
    names = sympy.numbered_symbols('t', exclude=set().union(*(e.free_symbols for e in exprs)))
    pairs, outs = sympy.cse(exprs, symbols=names)
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
        return _sympy_body(exprs)
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


def _choose_tiles(namespace, funcname, plan, values, n_out, extents):
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
    span = max(n_out, *(op.slots for op in plan if op.array))
    wide = math.prod(extents) * span >= 2 ** 31
    sizes = [f'e{k}' for k in range(1, len(extents))]

    def autotune(kernel, tiles, key, **kwargs):
        configs = [triton.Config(_sizes(shape), num_warps=w, num_stages=s) for shape, w, s in tiles]
        return triton.autotune(configs=configs, key=key, **kwargs)(kernel)

    widths = _widths(extents)
    clean = _widest_clean(namespace[f'{funcname}_fwd'], _tiles(widths, CONFIGS), *call, out, *extents, wide)
    namespace[f'{funcname}_fwd'] = autotune(namespace[f'{funcname}_fwd'], clean, sizes)

    elements = [(math.prod(shape), w, s) for shape, w, s in clean]
    candidates = _tiles(widths, elements, [1 << k for k in range(max(e for e, _, _ in elements).bit_length())])
    arrays = [(v, op) for v, op in zip(values, plan) if op.array]
    grads = {f'd{op.name}': v if all(op.varies) else v.to(torch.promote_types(v.dtype, torch.float32)) for v, op in arrays}
    flags = {f'G{op.name}': True for _, op in arrays}
    stripes = _stripes(extents, candidates, reference.device)
    namespace['_BACKWARD'] = _affordable(namespace[f'{funcname}_bwd'], candidates, reference.device, *call, *extents, *grads.values(), out, stripes, WIDE=wide, **flags)
    # Gradients are summed by atomic adds, so every timed run has to start them from zero again.
    namespace[f'{funcname}_bwd'] = autotune(namespace[f'{funcname}_bwd'], namespace['_BACKWARD'], sizes + list(flags), reset_to_zero=[f'd{op.name}' for _, op in arrays if not all(op.varies)])


def triton_lambdify(args, exprs, funcname, cse=True, output_mv_idx=None, values_asarray=None, shapes=None, printer=None):
    """
    A differentiable callable over stacked coefficient tensors, backed by a Triton kernel.

    :param shapes: ``{argument name: shape}``, from :func:`~kingdon.codegen.do_compile_symbolic`.
    :param printer: what :func:`~kingdon.codegen.lambdify` prints sympy expressions with, where there is no kernel.
    """
    from kingdon.codegen import lambdify

    plain = lambdify(args, exprs, funcname, printer=printer, cse=cse, output_mv_idx=output_mv_idx, values_asarray=values_asarray)
    try:
        if output_mv_idx is not None:
            raise Unsupported('writes into an argument')
        lines, outs = _body(exprs)
    except Unsupported:
        return plain

    import torch

    built = {}
    bases = []
    for name, vals in args.items():
        nested = any(isinstance(v, (list, tuple)) for v in vals)
        bases.append(Operand(name, tuple(_var(v) for v in (vals[0] if nested else vals)), nested))

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
        plan, extents = _plan(bases, tuple(datashape(value, base) for value, base in zip(values, bases)))
        dtype = functools.reduce(torch.promote_types,
                                 [v.dtype for v, op in zip(values, plan) if op.array])
        grad_lines, grads = _gradients(plan, exprs)
        src = _source(funcname, plan, len(extents), lines, outs, grad_lines, grads, dtype)
        filename = f'{funcname}#{next(_BUILDS)}'
        namespace = {'_layout': _layout, '_stripes': _stripes}
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        exec(compile(src, filename, 'exec'), namespace)
        _choose_tiles(namespace, funcname, plan, values, len(outs), extents)
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


def _source(funcname, plan, n, lines, outs, grad_lines, grads, dtype):
    n_out = len(outs)
    arrays = [op for op in plan if op.array]
    names = [op.name for op in arrays]
    flags = [f'G{name}' for name in names]
    span = max(n_out, *(op.slots for op in arrays))
    # An operand shared along the outermost axis -- a weight, shared over the batch -- sums its gradient into one of several copies, picked by block and added up by the
    # launcher, so that the blocks along that axis do not all contend for the same one.
    shared = any(not op.varies[0] for op in arrays)
    full, tile = (True,) * n, f'[{", ".join(f"T{k}" for k in range(n))}]'
    sizes = ", ".join(f"e{k}" for k in range(n))
    # Every constexpr follows every runtime argument: inductor launches a user kernel without its constexprs, yet finds the gradients to zero between timed configs by
    # their position in the whole signature. WIDE precedes the tile so that :func:`_widest_clean` can pass it positionally.
    constexprs = f'WIDE: tl.constexpr, {", ".join(f"T{k}: tl.constexpr" for k in range(n))}'
    index = [*_blocks(n), *_address(n, dict.fromkeys([full, *(op.varies for op in arrays)]))]

    fwd = ['@triton.jit',
           f'def {funcname}_fwd({", ".join(_params(plan))}, out, {sizes}, {constexprs}):',
           *index, *_loads(arrays), *lines,
           *(f'    tl.store(out + {_at(full, k)}, {e}, mask=_m_{_tag(full)})' for k, e in enumerate(outs))]

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

    bwd = ['@triton.jit',
           f'def {funcname}_bwd({", ".join(_params(plan))}, {sizes}, {", ".join("d" + name for name in names)}, gout, STRIPES, '
           f'{constexprs}, {", ".join(f"{f}: tl.constexpr" for f in flags)}):',
           *index, *_loads(arrays),
           *(f'    go{k} = tl.load(gout + {_at(full, k)}, mask=_m_{_tag(full)})' for k in range(n_out)),
           *grad_lines, *stores]

    shapes = ', '.join(f'{op.name}.shape[{op.lead}:]' for op in arrays)
    buffers = []
    for op, flag in zip(arrays, flags):
        # A gradient gathered from several blocks starts at zero and accumulates in at least single precision.
        make = (f'torch.empty_like({op.name})' if all(op.varies) else
                f'torch.zeros(({"" if op.varies[0] else "stripes, "}*{op.name}.shape,), device={op.name}.device, dtype=torch.promote_types({op.name}.dtype, torch.float32))')
        buffers.append(f'd{op.name} = {make} if {flag} else {op.name}.new_empty(0)')
    scalars = [var for op in plan if not op.array for var in op.vars]
    returns = ', '.join('None' if not op.array else f'(d{op.name}{"" if op.varies[0] else ".sum(0)"}.to({op.name}.dtype) if G{op.name} else None)' for op in plan)

    launcher = f"""
def _grid(meta):
    return ({" * ".join(f'triton.cdiv(meta["e{k}"], meta["T{k}"])' for k in range(n))},)


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {", ".join(op.name for op in plan)}):
        {"; ".join(f"{name} = {name}.contiguous()" for name in names)}
        {"; ".join(f"{', '.join(op.vars)}, = {op.name}" for op in plan if not op.array) or "pass"}
        data, extents, _ = _layout(({shapes},))
        out = torch.empty(({n_out}, *data), device={names[0]}.device, dtype={dtype})
        ctx.save_for_backward({", ".join(names)})
        ctx.scalars, ctx.extents = ({", ".join(scalars)}{"," if scalars else ""}), extents
        {funcname}_fwd[_grid]({", ".join(_params(plan))}, out, *extents, WIDE=math.prod(extents) * {span} >= 2 ** 31)
        return out

    @staticmethod
    def backward(ctx, gout):
        {", ".join(names)}, = ctx.saved_tensors
        ({", ".join(scalars)}{"," if scalars else ""}) = ctx.scalars
        {", ".join(flags)}, = {", ".join(f"ctx.needs_input_grad[{i}]" for i, op in enumerate(plan) if op.array)},
        gout, extents = gout.contiguous(), ctx.extents
        stripes = {"_stripes(extents, _BACKWARD, gout.device)" if shared else "1"}
        {"; ".join(buffers)}
        {funcname}_bwd[_grid]({", ".join(_params(plan))}, *extents, {", ".join(f"d{name}" for name in names)}, gout, stripes,
                              WIDE=math.prod(extents) * {span} >= 2 ** 31, {", ".join(f"{f}={f}" for f in flags)})
        return {returns}


def {funcname}(*values):
    return _Fn.apply(*values)
"""
    return ('import math\nimport torch\nimport triton\nimport triton.language as tl\n\n\n'
            + '\n'.join(fwd) + '\n\n\n' + '\n'.join(bwd) + '\n\n' + launcher)
