"""
One Triton kernel per operator, instead of one torch call per symbolic multiply::

    >>> alg = Algebra(3, lambdifier=triton_lambdify)
    >>> z = alg.gp(x, y)

Multivector in, multivector out, differentiable, same as any other lambdifier. Non-polynomial
expressions fall back to :func:`~kingdon.codegen.lambdify`; the backward comes from
:meth:`~kingdon.polynomial.RationalPolynomial.diff`.
"""
from __future__ import annotations

import functools
import itertools
import linecache
import math
from dataclasses import dataclass, replace

from kingdon.polynomial import RationalPolynomial, poly_format, rational_cse, rp_var_name

#: Tiles to consider, as (rows, columns, warps, stages). How many registers a tile needs is
#: not worth predicting -- measured against a count of the live coefficients the ratio ranged
#: from 0.92 to 2.09 -- so :func:`_widest_clean` compiles them and reads the spills instead.
CONFIGS = [(1, 64, 8, 1), (1, 64, 4, 1), (2, 64, 4, 1), (4, 128, 16, 1), (4, 256, 16, 2)]

_BUILDS = itertools.count()


class Unsupported(Exception):
    """Raised when an expression cannot be emitted as a kernel, so the caller falls back."""


@dataclass(frozen=True)
class Operand:
    """One argument of an operator: its coefficients, and how the kernel addresses them."""

    name: str
    vars: tuple[str, ...]
    nested: bool
    kind: str = ''
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
        return self.kind != 'scalar'

    @property
    def live(self):
        return [(i, var) for i, var in enumerate(self.vars) if var != '_']


def _load(tile):
    return tile[0] * tile[1] / tile[2]


def _widest_clean(kernel, sample):
    """
    The widest tile this kernel compiles for without spilling, and every narrower one.

    Spilling costs far more than a wider tile wins, and where the threshold falls depends on
    the operator and the card, so it is compiled and measured rather than guessed. Narrower
    tiles give each thread fewer elements and so cannot need more registers, which is why the
    search can stop at the first clean one.
    """
    from triton.errors import TritonError

    ordered = sorted(CONFIGS, key=_load, reverse=True)
    for i, (bb, fb, warps, stages) in enumerate(ordered):
        try:
            compiled = kernel.warmup(*sample, BB=bb, FB=fb, num_warps=warps, num_stages=stages, grid=(1,))
            compiled._init_handles()
        except TritonError:
            continue
        if compiled.n_spills == 0:
            return ordered[i:]
    return ordered[-1:]


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


def _roles(shapes):
    """
    How each operand is addressed, from the shapes alone.

    However many axes a multivector's coefficients carry, they are contiguous, so the kernel
    reads each blade as a flat rows-by-columns plane. ``batch`` is indexed by both axes,
    ``feature`` only by the last and is broadcast along the rows, ``scalar`` -- the
    ``math.sqrt(2)`` a layer divides by -- rides along by value, and ``broadcast`` is anything
    whose replication the flat offset cannot express, which the launcher spreads over the plane
    and whose gradient sums back down.

    Operands are right-aligned and broadcast against each other the way torch would, so a weight
    is recognised by the axes it varies along rather than by its rank: ``(1, 32)`` and ``(32,)``
    both address the last axis alone. The call path asks this too, to tell a shape the generated
    code already covers from one that needs its own.
    """
    plane = _plane(shapes)
    roles = []
    for shape in shapes:
        if shape is None:
            roles.append('scalar')
            continue
        aligned = (1,) * (len(plane) - len(shape)) + tuple(shape)
        if aligned == plane:
            roles.append('batch')
        elif aligned[-1] == plane[-1] and all(extent == 1 for extent in aligned[:-1]):
            roles.append('feature')
        else:
            roles.append('broadcast')
    return tuple(roles)


def _plan(bases, shapes):
    return [replace(base, kind=kind, shape=shape)
            for base, shape, kind in zip(bases, shapes, _roles(shapes))]


def _body(exprs):
    """CSE'd assignment lines and one formatted expression per output."""
    exprs = list(exprs)
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


def _loads(plan):
    return [f'    {var} = tl.load({op.name} + {i} * feat + fi, mask=fmask)[None, :]'
            if op.kind == 'feature' else
            f'    {var} = tl.load({op.name} + {i} * plane + off, mask=mask)'
            for op in plan if op.array for i, var in op.live]


def _params(plan):
    """Kernel parameters: a pointer per array argument, a value per scalar one."""
    return [var for op in plan for var in (op.vars if not op.array else [op.name])]


def _call(plan):
    return [var if not op.array else
            (f'_spread({op.name}, data, {op.lead})' if op.kind == 'broadcast' else op.name)
            for op in plan for var in (op.vars if not op.array else [op.name])]


_HEADER = """
    bi = tl.program_id(0) * BB + tl.arange(0, BB)
    fi = tl.max_contiguous(tl.multiple_of(tl.program_id(1) * FB + tl.arange(0, FB), FB), FB)
    if WIDE:
        bi, fi = bi.to(tl.int64), fi.to(tl.int64)
    fmask = fi < feat
    mask = (bi < batch)[:, None] & fmask[None, :]
    off = bi[:, None] * feat + fi[None, :]"""


def _choose_tiles(namespace, funcname, plan, values, n_out):
    """Autotune the forward over the tiles that do not spill, and give the backward the widest."""
    import torch
    import triton

    data = _plane([op.shape for op in plan])
    feat, batch = data[-1], math.prod(data[:-1])
    reference = next(v for v in values if torch.is_tensor(v))
    call = [namespace['_spread'](v, data, op.lead) if op.kind == 'broadcast' else v if op.array else v[0]
            for v, op in zip(values, plan)]
    out = torch.empty((n_out, *data), device=reference.device, dtype=reference.dtype)
    span = max(n_out, *(op.slots for op in plan if op.array))
    sample = (*call, out, batch, feat, batch * feat, batch * feat * span >= 2 ** 31)

    clean = _widest_clean(namespace[f'{funcname}_fwd'], sample)
    # Narrowing a tile to the data does not pay: the grid is unchanged, and a tile smaller than
    # the warps it was tuned for leaves most of them idle. Measured on o3, whose first layer has
    # three columns, clamping 256 down to 4 cost 10%. The autotuner picks per feat instead.
    namespace[f'{funcname}_fwd'] = triton.autotune(
        configs=[triton.Config({'BB': bb, 'FB': fb}, num_warps=w, num_stages=s)
                 for bb, fb, w, s in clean],
        key=['feat'])(namespace[f'{funcname}_fwd'])

    # The backward takes whichever tile the forward timed fastest, read at call time. Nothing
    # is spill-free for a large algebra there, and a wide tile that spills beats a narrow one
    # that does not. A stale choice only costs speed, never correctness.
    namespace['_FALLBACK'] = clean[0]


def triton_lambdify(args, exprs, funcname, cse=True, output_mv_idx=None, values_asarray=None, shapes=None):
    """
    A differentiable callable over stacked coefficient tensors, backed by a Triton kernel.

    :param shapes: ``{argument name: shape}``, from :func:`~kingdon.codegen.do_compile_symbolic`.
    """
    from kingdon.codegen import lambdify

    plain = lambdify(args, exprs, funcname, cse=cse, output_mv_idx=output_mv_idx, values_asarray=values_asarray)
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
        bases.append(Operand(name, tuple(rp_var_name(v) for v in (vals[0] if nested else vals)), nested))

    def datashape(value, base):
        return tuple(value.shape[base.lead:]) if torch.is_tensor(value) else None

    def signature(values):
        """
        Everything the generated code depends on: how each operand is addressed, how many
        coefficients it carries, and its dtype. Deliberately not the extents -- the kernel takes
        those as arguments, so one build serves every size.

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
            return tuple(zip(_roles(shapes_in), dtypes))
        except Unsupported:
            return None

    def build(values):
        """
        Emit the kernel from the roles the operands actually arrive with.

        `shapes` cannot be trusted for this. An operator is cached on (type, keys), and the
        first call that creates it is often the symbolic one inside another operator's
        codegen, where every multivector is shapeless -- so whether an operator got a kernel
        would depend on the order codegen happened to reach it in.
        """
        plan = _plan(bases, [datashape(value, base) for value, base in zip(values, bases)])
        dtype = functools.reduce(torch.promote_types,
                                 [v.dtype for v, op in zip(values, plan) if op.array])
        grad_lines, grads = _gradients(plan, exprs)
        src = _source(funcname, plan, lines, outs, grad_lines, grads, dtype)
        filename = f'{funcname}#{next(_BUILDS)}'
        namespace = {}
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        exec(compile(src, filename, 'exec'), namespace)
        _choose_tiles(namespace, funcname, plan, values, len(outs))
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
    go = [RationalPolynomial.fromname(f'go{k}') for k in range(len(exprs))]
    loss = go[0] * exprs[0]
    for g, e in zip(go[1:], exprs[1:]):
        loss = loss + g * e
    names = [var for op in plan if op.array for _, var in op.live]
    lines, formatted = _body([loss.diff(var) for var in names])
    return lines, dict(zip(names, formatted))


def _source(funcname, plan, lines, outs, grad_lines, grads, dtype):
    n_out = len(outs)
    arrays = [op for op in plan if op.array]
    names = [op.name for op in arrays]
    flags = [f'G{op.name}' for op in arrays]
    span = max(n_out, *(op.slots for op in arrays))
    # WIDE precedes the tile so that :func:`_widest_clean` can pass it positionally.
    tail = 'batch, feat, plane, WIDE: tl.constexpr, BB: tl.constexpr, FB: tl.constexpr'

    fwd = ['@triton.jit',
           f'def {funcname}_fwd({", ".join(_params(plan))}, out, {tail}):', _HEADER.strip('\n'),
           *_loads(plan), *lines,
           *(f'    tl.store(out + {k} * plane + off, {e}, mask=mask)' for k, e in enumerate(outs))]

    stores = []
    for op in arrays:
        stores.append(f'    if G{op.name}:')
        for i, var in op.live:
            if op.kind == 'feature':
                # Padding rows are summed over too, so they are zeroed rather than trusted to
                # evaluate to zero: a derivative with a denominator is 0/0 on a masked lane.
                stores.append(f'        tl.store(d{op.name} + tl.program_id(0) * {op.slots} * feat '
                              f'+ {i} * feat + fi, tl.sum(tl.where(mask, {grads[var]}, 0.0), axis=0), mask=fmask)')
            else:
                stores.append(f'        tl.store(d{op.name} + {i} * plane + off, {grads[var]}, mask=mask)')

    bwd = ['@triton.jit',
           f'def {funcname}_bwd({", ".join(_params(plan))}, {tail}, '
           f'{", ".join("d" + n for n in names)}, gout, '
           f'{", ".join(f"{f}: tl.constexpr" for f in flags)}):',
           _HEADER.strip('\n'), *_loads(plan),
           *(f'    go{k} = tl.load(gout + {k} * plane + off, mask=mask)' for k in range(n_out)),
           *grad_lines, *stores]

    extents = ', '.join(f'{op.name}.shape[{op.lead}:]' for op in arrays)
    buffers, folds = [], []
    for op, flag in zip(arrays, flags):
        if op.kind == 'feature':
            buffers.append(f'd{op.name} = torch.empty((blocks, {op.slots}, feat) if {flag} else (0,), device={op.name}.device, dtype={op.name}.dtype)')
            folds.append(f'd{op.name} = d{op.name}.sum(0).reshape({op.name}.shape) if {flag} else None')
        else:
            buffers.append(f'd{op.name} = torch.empty((*{op.name}.shape[:{op.lead}], *data) if {flag} else (0,), device={op.name}.device, dtype={op.name}.dtype)')
            folds.append(f'd{op.name} = _reduce(d{op.name}, {op.name}, {op.lead}) if {flag} else None')
    scalars = [var for op in plan if not op.array for var in op.vars]

    launcher = f'''
def _grid(meta):
    return (triton.cdiv(meta["batch"], meta["BB"]), triton.cdiv(meta["feat"], meta["FB"]))


def _extents(*shapes):
    data = torch.broadcast_shapes(*shapes)
    return data, math.prod(data[:-1]), data[-1]


def _spread(operand, data, lead):
    """Give a broadcast operand the whole plane, so the kernel reads it like any other."""
    operand = operand.reshape(*operand.shape[:lead], *(1,) * (len(data) + lead - operand.ndim), *operand.shape[lead:])
    return operand.expand(*operand.shape[:lead], *data).contiguous()


def _reduce(grad, like, lead):
    """Sum a whole-plane gradient back onto the shape the operand actually had."""
    target = (*like.shape[:lead], *(1,) * (grad.ndim - like.ndim), *like.shape[lead:])
    return grad.sum_to_size(target).reshape(like.shape)


def _tile():
    best = getattr({funcname}_fwd, 'best_config', None)
    return (best.kwargs['BB'], best.kwargs['FB'], best.num_warps, best.num_stages) if best else _FALLBACK


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {", ".join(op.name for op in plan)}):
        {"; ".join(f"{n} = {n}.contiguous()" for n in names)}
        {"; ".join(f"{', '.join(op.vars)}, = {op.name}" for op in plan if not op.array) or "pass"}
        data, batch, feat = _extents({extents})
        out = torch.empty(({n_out}, *data), device={names[0]}.device, dtype={dtype})
        ctx.save_for_backward({", ".join(names)})
        ctx.scalars = ({", ".join(scalars)}{"," if scalars else ""})
        {funcname}_fwd[_grid]({", ".join(_call(plan))}, out, batch=batch, feat=feat,
            plane=batch * feat, WIDE=batch * feat * {span} >= 2 ** 31)
        return out

    @staticmethod
    def backward(ctx, gout):
        {", ".join(names)}, = ctx.saved_tensors
        ({", ".join(scalars)}{"," if scalars else ""}) = ctx.scalars
        {", ".join(flags)}, = {", ".join(f"ctx.needs_input_grad[{i}]" for i, op in enumerate(plan) if op.array)},
        gout = gout.contiguous()
        data, batch, feat = _extents({extents})
        BB, FB, warps, stages = _tile()
        blocks = triton.cdiv(batch, BB)
        {"; ".join(buffers)}
        {funcname}_bwd[(blocks, triton.cdiv(feat, FB))](
            {", ".join(_call(plan))}, batch, feat, batch * feat,
            batch * feat * {span} >= 2 ** 31, BB, FB,
            {", ".join("d" + n for n in names)}, gout,
            {", ".join(flags)}, num_warps=warps, num_stages=stages)
        {"; ".join(folds)}
        return {", ".join(('None' if not op.array else 'd' + op.name) for op in plan)}


def {funcname}(*values):
    return _Fn.apply(*values)
'''
    return ('import math\nimport torch\nimport triton\nimport triton.language as tl\n\n\n'
            + '\n'.join(fwd) + '\n\n\n' + '\n'.join(bwd) + '\n\n' + launcher)
