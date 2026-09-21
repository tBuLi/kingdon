"""
One Triton kernel per operator, instead of one torch call per symbolic multiply::

    >>> alg = Algebra(3, lambdifier=triton_lambdify)
    >>> z = alg.gp(x, y)

Multivector in, multivector out, differentiable, same as any other lambdifier. Non-polynomial
expressions fall back to :func:`~kingdon.codegen.lambdify`; the backward comes from
:meth:`~kingdon.polynomial.RationalPolynomial.diff`.
"""
from __future__ import annotations

import linecache

from kingdon.polynomial import RationalPolynomial, poly_format, rational_cse, rp_var_name

#: Tiles to consider, as (rows, columns, warps, stages). How many registers a tile needs is
#: not worth predicting -- measured against a count of the live coefficients the ratio ranged
#: from 0.92 to 2.09 -- so :func:`_widest_clean` compiles them and reads the spills instead.
CONFIGS = [(1, 64, 8, 1), (1, 64, 4, 1), (2, 64, 4, 1), (4, 128, 16, 1), (4, 256, 16, 2)]


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
    ordered = sorted(CONFIGS, key=_load, reverse=True)
    for i, (bb, fb, warps, stages) in enumerate(ordered):
        try:
            compiled = kernel.warmup(*sample, BB=bb, FB=fb, num_warps=warps, num_stages=stages, grid=(1,))
            compiled._init_handles()
        except Exception:
            continue
        if compiled.n_spills == 0:
            return ordered[i:]
    return ordered[-1:]


class Unsupported(Exception):
    """Raised when an expression cannot be emitted as a kernel, so the caller falls back."""


def _plan(args, datashapes):
    """
    Per argument: its variable names, how it is addressed, and the data shape it must arrive with.

    However many axes a multivector's coefficients carry, they are contiguous, so the kernel
    reads each blade as a flat rows-by-columns plane: the last axis is the one a weight is
    shared along, and everything before it is rows. An operand with the full data is indexed by
    both, one with just that last axis is broadcast along the rows, and a shapeless one -- the
    ``math.sqrt(2)`` a layer divides by -- rides along by value.
    """
    full = max(datashapes.values(), key=_numel)
    if not full:
        raise Unsupported('nothing to tile over')
    feat = full[-1]
    plan = []
    for name, values in args.items():
        shape = datashapes[name]
        nested = any(isinstance(v, (list, tuple)) for v in values)
        flat = list(values[0]) if nested else list(values)
        if not shape:
            kind = 'scalar'
        elif shape == full:
            kind = 'batch'
        elif shape == (feat,):
            kind = 'feature'
        else:
            raise Unsupported(f'{name} is {shape}, neither {full} nor its last axis')
        plan.append((name, [rp_var_name(v) for v in flat], kind, shape))
    return plan


def _numel(shape):
    total = 1
    for extent in shape:
        total *= extent
    return total


def _cover(extent):
    """The smallest power of two that spans the extent, since a tile axis is a ``tl.arange`` and has to be one."""
    return 1 << (extent - 1).bit_length()


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


def _loads(plan, indent='    '):
    lines = []
    for name, names, kind, _ in plan:
        for i, var in enumerate(names):
            if var == '_' or kind == 'scalar':
                continue
            if kind == 'batch':
                lines.append(f'{indent}{var} = tl.load({name} + {i} * plane + off, mask=mask)')
            else:
                lines.append(f'{indent}{var} = tl.load({name} + {i} * feat + fi, '
                             f'mask=fmask)[None, :]')
    return lines


def _params(plan):
    """Kernel parameters: a pointer per array argument, a value per scalar one."""
    return [var if kind == 'scalar' else name
            for name, names, kind, _ in plan for var in (names if kind == 'scalar' else [name])]


_HEADER = """
    bi = tl.program_id(0) * BB + tl.arange(0, BB)
    fi = tl.program_id(1) * FB + tl.arange(0, FB)
    bmask = bi < batch
    fmask = fi < feat
    mask = bmask[:, None] & fmask[None, :]
    off = bi[:, None] * feat + fi[None, :]
    plane = batch * feat"""


def _choose_tiles(namespace, funcname, plan, values, n_out):
    """Autotune the forward over the tiles that do not spill, and give the backward the widest."""
    import torch
    import triton

    data = next(shape for _, _, kind, shape in plan if kind == 'batch')
    feat, batch = data[-1], _numel(data) // data[-1]
    reference = next(v for v in values if torch.is_tensor(v))
    call = [v if torch.is_tensor(v) else v[0] for v in values]
    out = torch.empty((n_out, *data), device=reference.device, dtype=reference.dtype)

    clean = _widest_clean(namespace[f'{funcname}_fwd'], (*call, out, batch, feat))
    namespace[f'{funcname}_fwd'] = triton.autotune(
        configs=[triton.Config({'BB': bb, 'FB': fb}, num_warps=w, num_stages=s)
                 for bb, fb, w, s in clean],
        key=['batch', 'feat'])(namespace[f'{funcname}_fwd'])

    # The backward takes whichever tile the forward timed fastest, read at call time. Nothing
    # is spill-free for a large algebra there, and a wide tile that spills beats a narrow one
    # that does not. A stale choice only costs speed: the launcher sizes the weight-gradient
    # buffer from whatever it reads and passes that same tile to the kernel.
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

    def build(values):
        """
        Emit the kernel from the ranks the operands actually arrive with.

        `shapes` cannot be trusted for this. An operator is cached on (type, keys), and the
        first call that creates it is often the symbolic one inside another operator's
        codegen, where every multivector is shapeless -- so whether an operator got a kernel
        would depend on the order codegen happened to reach it in.
        """
        plan = _plan(args, {name: (tuple(value.shape[2 if any(isinstance(v, (list, tuple)) for v in vals) else 1:])
                                   if torch.is_tensor(value) else ())
                            for (name, vals), value in zip(args.items(), values)})
        grad_lines, grads = _gradients(plan, exprs)
        ptrs = [name for name, _, _, _ in plan]
        src = _source(funcname, plan, ptrs, lines, outs, grad_lines, grads)
        namespace = {}
        linecache.cache[funcname] = (len(src), None, src.splitlines(True), funcname)
        exec(compile(src, funcname, 'exec'), namespace)
        _choose_tiles(namespace, funcname, plan, values, len(outs))
        return namespace[funcname], namespace, src, [(kind, shape, len(names), any(isinstance(v, (list, tuple)) for v in args[name])) for name, names, kind, shape in plan]

    def fits(values, wanted):
        """
        Every operand has to arrive with the shape the kernel was built for. Comparing ranks
        would not do: the offsets come from one operand's extents, so a multivector torch
        would broadcast would be read as though it had the rows it does not have.
        """
        if len(values) != len(wanted):
            return False
        for value, (kind, shape, slots, nested) in zip(values, wanted):
            # A scalar has to be plain numbers, not a zero-dimensional tensor: the kernel
            # takes it by value and reports no gradient, which only holds for a constant.
            if kind == 'scalar':
                if not (isinstance(value, (list, tuple)) and len(value) == slots and not any(torch.is_tensor(v) for v in value)):
                    return False
            elif not torch.is_tensor(value) or not value.is_cuda or tuple(value.shape) != ((1, slots, *shape) if nested else (slots, *shape)):
                return False
        return True

    def dispatch(*values):
        entry = built.get('entry')
        if entry is None:
            # Symbolic calls come through here too, during another operator's codegen, and
            # carry polynomials rather than tensors. Build from nothing and every argument
            # looks shapeless, so wait for real ones rather than remembering the failure.
            if not any(torch.is_tensor(v) and v.is_cuda for v in values):
                return plain(*values)
            try:
                entry = built['entry'] = build(values)
                dispatch.source, dispatch.kernels = entry[2], entry[1]
            except Unsupported:
                built['entry'] = False
                return plain(*values)
        if entry is False:
            return plain(*values)
        kernel, _, _, wanted = entry
        return kernel(*values) if fits(values, wanted) else plain(*values)

    dispatch.__name__ = funcname
    dispatch.source = None  # set once the operands show what the kernel has to address
    dispatch.kernels = {}
    return dispatch


def _gradients(plan, exprs):
    """d(sum_k go_k * out_k)/ds per input symbol, CSE'd together so they share work."""
    go = [RationalPolynomial.fromname(f'go{k}') for k in range(len(exprs))]
    loss = go[0] * exprs[0]
    for g, e in zip(go[1:], exprs[1:]):
        loss = loss + g * e
    names = [var for _, vars_, kind, _ in plan if kind != 'scalar'
             for var in vars_ if var != '_']
    lines, formatted = _body([loss.diff(var) for var in names])
    return lines, dict(zip(names, formatted))


def _source(funcname, plan, ptrs, lines, outs, grad_lines, grads):
    n_out = len(outs)
    out_grad = [f'go{k}' for k in range(n_out)]
    params = _params(plan)
    arrays = [name for name, _, kind, _ in plan if kind != 'scalar']

    fwd = ['@triton.jit',
           f'def {funcname}_fwd({", ".join(params)}, out, batch, feat, '
           'BB: tl.constexpr, FB: tl.constexpr):', _HEADER.strip('\n'),
           *_loads(plan), *lines,
           *(f'    tl.store(out + {k} * plane + off, {e}, mask=mask)'
             for k, e in enumerate(outs))]

    # A weight is shared by every row, so its gradient sums over the batch: each block owns a buffer slice and the launcher adds them, no atomics.
    stores = []
    for name, names, kind, _ in plan:
        if kind == 'scalar':
            continue  # a python number is a constant, and has no gradient to store
        for i, var in enumerate(names):
            if var == '_':
                continue
            if kind == 'batch':
                stores.append(f'    tl.store(d{name} + {i} * plane + off, {grads[var]}, '
                              f'mask=mask)')
            else:
                stores.append(f'    tl.store(d{name} + tl.program_id(0) * {len(names)} * feat '
                              f'+ {i} * feat + fi, tl.sum({grads[var]}, axis=0), mask=fmask)')

    bwd = ['@triton.jit',
           f'def {funcname}_bwd({", ".join(params)}, '
           f'{", ".join("d" + p for p in arrays)}, '
           'gout, batch, feat, BB: tl.constexpr, FB: tl.constexpr):', _HEADER.strip('\n'),
           *_loads(plan),
           *(f'    {g} = tl.load(gout + {k} * plane + off, mask=mask)'
             for k, g in enumerate(out_grad)),
           *grad_lines, *stores]

    batched_arg = next(name for name, _, kind, _ in plan if kind == 'batch')
    data_shape = next(shape for _, _, kind, shape in plan if kind == 'batch')
    feat = data_shape[-1]
    batch = _numel(data_shape) // feat
    weights = [(name, len(names)) for name, names, kind, _ in plan if kind == 'feature']
    saves = ', '.join(ptrs)
    call = _params(plan)
    scalars = [var for _, names, kind, _ in plan if kind == 'scalar' for var in names]
    # A scalar is a plain number, so it has no gradient to give back.
    scalar_unpack = '; '.join(
        f'{", ".join(names)}, = {name}' for name, names, kind, _ in plan if kind == 'scalar')
    returns = [('None' if kind == 'scalar' else 'd' + name)
               for name, _, kind, _ in plan]

    launcher = f'''
def _grid(meta):
    return (triton.cdiv(meta["batch"], meta["BB"]), triton.cdiv(meta["feat"], meta["FB"]))


def _tile():
    best = getattr({funcname}_fwd, 'best_config', None)
    return (best.kwargs['BB'], best.kwargs['FB'], best.num_warps, best.num_stages) if best else _FALLBACK


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {saves}):
        {"; ".join(f"{p} = {p}.contiguous()" for p in arrays)}
        {scalar_unpack if scalars else "pass"}
        batch, feat = {batch}, {feat}
        out = torch.empty(({n_out}, {", ".join(str(d) for d in data_shape)}), device={batched_arg}.device, dtype={batched_arg}.dtype)
        {funcname}_fwd[_grid]({", ".join(call)}, out, batch=batch, feat=feat)
        ctx.save_for_backward({", ".join(arrays)})
        ctx.scalars = ({", ".join(scalars)}{"," if scalars else ""})
        return out

    @staticmethod
    def backward(ctx, gout):
        {", ".join(arrays)}, = ctx.saved_tensors
        ({", ".join(scalars)}{"," if scalars else ""}) = ctx.scalars
        gout = gout.contiguous()
        batch, feat = {batch}, {feat}
        tile_bb, tile_fb, warps, stages = _tile()
        BB, FB = min(tile_bb, {_cover(batch)}), min(tile_fb, {_cover(feat)})
        blocks = triton.cdiv(batch, BB)
        {"; ".join(f"d{p} = torch.zeros_like({p})" for p, _, k, _ in plan if k == "batch")}
        {"; ".join(f"d{p} = torch.empty((blocks, {k}, feat), device={batched_arg}.device, dtype={batched_arg}.dtype)" for p, k in weights)}
        {funcname}_bwd[(blocks, triton.cdiv(feat, FB))](
            {", ".join(call)}, {", ".join("d" + p for p in arrays)}, gout, batch, feat, BB, FB,
            num_warps=warps, num_stages=stages)
        {"; ".join(f"d{p} = d{p}.sum(0).reshape({p}.shape)" for p, _ in weights)}
        return {", ".join(returns)}


def {funcname}(*values):
    return _Fn.apply(*values)
'''
    return ('import torch\nimport triton\nimport triton.language as tl\n\n\n'
            + '\n'.join(fwd) + '\n\n\n' + '\n'.join(bwd) + '\n\n' + launcher)
