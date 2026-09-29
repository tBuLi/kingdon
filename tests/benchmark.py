"""
Benchmark kingdon's algebra creation and code generation.

Operator and expression rows are measured twice:

* ``cold_ms``  -- the very first call, which is what triggers codegen. Because a
  given Algebra can only be cold once, each sample builds a *fresh* Algebra and
  operands (untimed) and then times only the call. Reported as the min over up to
  7 such samples, with the median alongside; rows that are expensive enough to
  blow the time budget are measured once and say so in ``cold_n``.
* ``hot_us``   -- the steady-state cost of the same call once the generated
  function is cached, i.e. the numerical performance of the generated code.
  Measured with ``timeit.repeat``: a batch size chosen from a pilot run, then 5
  batches, reporting the best per-call time.

Construction rows only have ``hot_us``. Building an Algebra does no codegen, so
it is the same work every time and there is no cold/hot distinction to draw.

Min rather than mean throughout, since scheduler noise only ever adds time.
``*_med_*`` and ``*_n`` columns are there so you can see the spread and sample
count rather than taking the headline on trust.

Usage::

    python tests/benchmark.py
    python tests/benchmark.py --out results.csv --algebras 2DPGA 3DPGA 3DVGA
    python tests/benchmark.py --algebras 2DPGA --sections operator
    python tests/benchmark.py --skip outertan:multivector

The results are written to a CSV file (``tests/benchmark_results.csv`` by
default) which ``tests/benchmark_report.html`` renders as an interactive page.

Run it on an otherwise idle machine: these are wall-clock timings, and a busy
browser in the background is enough to roughly double every number in the table.
"""
import argparse
import csv
import os
import random
import re
import statistics
import sys
import time
import timeit
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kingdon import Algebra
from kingdon.operator_dict import UnaryOperatorDict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, 'benchmark_results.csv')

FIELDS = ['section', 'algebra', 'd', 'operation', 'variant', 'inputs',
          'out_keys', 'muls', 'divs', 'adds',
          'cold_ms', 'cold_med_ms', 'cold_n',      # codegen; blank for construction rows
          'hot_us', 'hot_med_us', 'hot_n',         # cached call; for construction, the construction itself
          'note']

ALGEBRAS = {
    '2DVGA': lambda **kw: Algebra(2, **kw),
    '3DVGA': lambda **kw: Algebra(3, **kw),
    '4DVGA': lambda **kw: Algebra(4, **kw),
    '2DPGA': lambda **kw: Algebra.fromname('2DPGA', **kw),
    '3DPGA': lambda **kw: Algebra.fromname('3DPGA', **kw),
    'STA':   lambda **kw: Algebra(3, 1, **kw),
    'STAP':  lambda **kw: Algebra.fromname('STAP', **kw),
    '3DCGA': lambda **kw: Algebra(4, 1, **kw),
}

# How an expression gets compiled, in the order they are drawn.
EXPR_VARIANTS = [
    ('traced',         dict(symbolic=False)),
    ('symbolic',       dict(symbolic=True)),
    ('symbolic-nocse', dict(symbolic=True, cse=False)),
    ('plain',          dict(plain=True)),
]

_COUNT_RE = re.compile(r'(\d+)\s*muls\s*/\s*(\d+)\s*divs\s*/\s*(\d+)\s*adds')


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def warmup():
    """
    Touch the lazily initialised machinery once.

    The very first geometric operation in a process pays a one-off cost (tens of
    ms) for imports and module-level setup that has nothing to do with codegen.
    Without this, whichever row happens to run first reports a wildly inflated
    ``first_ms``.
    """
    alg = Algebra.fromname('2DPGA')
    p, q = alg.point([1.0, 2.0]), alg.point([3.0, 4.0])
    v, w = alg.vector([1.0, 2.0, 3.0]), alg.vector([2.0, 3.0, 4.0])
    m = alg.multivector([1.0 + i for i in range(len(alg))])
    R = alg.gp(v, w)
    for call in (lambda: alg.gp(p, q), lambda: alg.div(v, w), lambda: alg.sw(R, p),
                 lambda: alg.inv(m), lambda: alg.sqrt(R), lambda: alg.normsq(m),
                 lambda: alg.outerexp(alg.bivector([1.0, 2.0, 3.0])),
                 lambda: alg.add_operator(symbolic=True, name='_warm')(lambda a, b: (a & b) * ~(a & b))(p, q)):
        try:
            call()
        except Exception:
            pass


def time_cold(run_once, budget=0.2, max_reps=7):
    """
    Time an operation that can only happen once per Algebra -- codegen.

    ``run_once`` must build a throwaway Algebra and its operands *untimed*, then
    time only the call, and return the elapsed seconds. We repeat that whole
    cycle so a cold measurement is a min over several independent samples rather
    than a single stopwatch reading. Expensive rows blow the budget on the first
    sample and are measured once, which is the best that can be done for them.

    :return: (min ms, median ms, samples)
    """
    times = []
    while True:
        times.append(run_once())
        if sum(times) >= budget or len(times) >= max_reps:
            break
    return min(times) * 1e3, statistics.median(times) * 1e3, len(times)


def time_hot(call, batch=0.03, repeats=5, max_number=2_000_000):
    """
    Steady-state cost of a repeatable call, the canonical ``timeit`` way: pick a
    batch size from a pilot run, then take ``repeats`` batches and report the
    best. Min rather than mean, because scheduler noise only ever adds time.

    :return: (min µs/call, median µs/call, total calls)
    """
    pilot = timeit.timeit(call, number=10) / 10
    number = max(1, min(max_number, int(batch / pilot))) if pilot > 0 else max_number
    per = [s / number for s in timeit.repeat(call, repeat=repeats, number=number)]
    return min(per) * 1e6, statistics.median(per) * 1e6, number * repeats


# ---------------------------------------------------------------------------
# Section 1: algebra creation
# ---------------------------------------------------------------------------

def bench_creation(rows, dims=range(2, 9)):
    """
    Building an Algebra is the same work every time -- there is no codegen step and
    so no cold/hot distinction. One properly repeated measurement is the whole story.

    Both storage modes are measured over the whole range, rather than letting the
    ``large = d > 6`` default pick one. That way the tipping point shows up as two
    separate trends the default hops between, instead of a kink in a single curve.
    """
    for d in dims:
        for label, build in (
            (f'Algebra({d})', (lambda large, d=d: Algebra(d, large=large))),
            (f'Algebra({d - 1},0,1)', (lambda large, d=d: Algebra(d - 1, 0, 1, large=large))),
        ):
            for mode, large in (('small', False), ('large', True)):
                make = (lambda build=build, large=large: build(large))
                best, med, runs = time_hot(make, batch=0.05)
                alg = make()
                rows.append(dict(section='creation', algebra=label, d=alg.d,
                                 operation='construct', variant=mode, inputs='',
                                 out_keys=len(alg), muls='', divs='', adds='',
                                 cold_ms='', cold_med_ms='', cold_n='',
                                 hot_us=round(best, 3), hot_med_us=round(med, 3), hot_n=runs,
                                 note='default' if large == (d > 6) else ''))
                print(f"  {label:<16} {mode:<6} {best / 1e3:8.4f} ms best"
                      f"   {med / 1e3:8.4f} ms median   (n={runs})"
                      f"{'   <- default' if large == (d > 6) else ''}", flush=True)


# ---------------------------------------------------------------------------
# Section 2: the standard operators
# ---------------------------------------------------------------------------

def numeric(alg, ctor):
    """A multivector of the requested type filled with random values."""
    n = len(getattr(alg, ctor)(name='_probe').keys())
    return getattr(alg, ctor)([random.uniform(0.5, 1.5) for _ in range(n)])


def operand_types(alg):
    types = ['vector', 'bivector', 'bireflection', 'multivector', 'point', 'translation']
    return [t for t in types if hasattr(alg, t)]


# The outer exponential family is only defined for bivectors -- it warns for anything
# mixed-grade, and on a dense multivector `outertan` spends tens of seconds compiling an
# expression that was never meaningful. Measure it where it is actually defined.
BIVECTOR_ONLY = {'outerexp', 'outersin', 'outercos', 'outertan'}


def operand_profiles(alg, arity, op_name=None):
    """
    Unary ops get every operand type; binary ops get the full cross product, so the
    report can offer "pick a type for each operand, see every product".
    """
    types = operand_types(alg)
    if op_name in BIVECTOR_ONLY:
        types = [t for t in types if t == 'bivector']
    if arity == 1:
        return [(t,) for t in types]
    return [(a, b) for a in types for b in types]


def op_counts(compiled):
    m = _COUNT_RE.search(compiled.func.__doc__ or '')
    return m.groups() if m else ('', '', '')


def bench_operators(rows, alg_name, factory, skip=()):
    probe = factory()
    print(f"\n  --- {alg_name} operators ---", flush=True)
    for op_name, opdict in probe.registry.items():
        arity = 1 if isinstance(opdict, UnaryOperatorDict) else 2
        for ctors in operand_profiles(probe, arity, op_name):
            if f"{op_name}:{'+'.join(ctors)}" in skip:
                print(f"  {op_name:<10} {'+'.join(ctors):<28} skipped", flush=True)
                continue
            state = {}

            def run_once():
                """Fresh Algebra + operands untimed; only the codegen-triggering call is timed."""
                alg = factory()
                mvs = [numeric(alg, c) for c in ctors]
                t0 = time.perf_counter()
                getattr(alg, op_name)(*mvs)
                dt = time.perf_counter() - t0
                state['alg'], state['mvs'] = alg, mvs
                return dt

            try:
                cold, cold_med, cold_n = time_cold(run_once)
            except Exception as e:
                rows.append(dict(section='operator', algebra=alg_name, d=probe.d,
                                 operation=op_name, variant='', inputs='+'.join(ctors),
                                 out_keys='', muls='', divs='', adds='',
                                 cold_ms='', cold_med_ms='', cold_n='',
                                 hot_us='', hot_med_us='', hot_n='',
                                 note=f'{type(e).__name__}: {e}'))
                print(f"  {op_name:<10} {'+'.join(ctors):<28} FAILED {type(e).__name__}", flush=True)
                continue

            alg, mvs = state['alg'], state['mvs']     # warm: generated function cached
            hot, hot_med, hot_n = time_hot(lambda: getattr(alg, op_name)(*mvs))
            compiled = getattr(alg, op_name)[mvs[0] if arity == 1 else tuple(mvs)]
            muls, divs, adds = op_counts(compiled)
            rows.append(dict(section='operator', algebra=alg_name, d=alg.d,
                             operation=op_name, variant='', inputs='+'.join(ctors),
                             out_keys=len(compiled.keys_out), muls=muls, divs=divs, adds=adds,
                             cold_ms=round(cold, 4), cold_med_ms=round(cold_med, 4), cold_n=cold_n,
                             hot_us=round(hot, 3), hot_med_us=round(hot_med, 3), hot_n=hot_n,
                             note=''))
            print(f"  {op_name:<10} {'+'.join(ctors):<28} {cold:9.3f} ms (n={cold_n})"
                  f"  {hot:8.2f} us", flush=True)


# ---------------------------------------------------------------------------
# Section 3: realistic added operator expressions
# ---------------------------------------------------------------------------

def project(x, y):
    """(x|y)/y -- project x onto y"""
    return (x | y) / y

def join_area(a, b, c):
    """(a&b&c)*~(a&b&c) -- squared area of a triangle"""
    return (a & b & c) * ~(a & b & c)

def join_volume(a, b, c, d):
    """a&b&c&d -- signed volume of a tetrahedron"""
    return a & b & c & d

def compose_and_apply(T, R, p):
    """(T*R)>>p -- compose two motors, then move a point"""
    return (T * R) >> p

def apply_twice(T, R, p):
    """T>>(R>>p) -- apply two motors in sequence"""
    return T >> (R >> p)

def normalize(M):
    """M/sqrt(M*~M) -- normalize a motor"""
    return M / M.normsq().sqrt()

def triple_sandwich(R, x):
    """R>>(R>>(R>>x)) -- three nested sandwiches"""
    return R >> (R >> (R >> x))

def chain(a, b, c, d, e):
    """a*b*c*d*e -- fivefold geometric product"""
    return a * b * c * d * e

def barycentric(A, B, C, P):
    """(P&B&C)/(A&B&C) -- ratio of two joins"""
    return (P & B & C) / (A & B & C)


PGA_EXPRS = [
    (project,          ('point', 'vector')),
    (join_area,        ('point', 'point', 'point')),
    (join_volume,      ('point', 'point', 'point', 'point')),
    (compose_and_apply,('translation', 'bireflection', 'point')),
    (apply_twice,      ('translation', 'bireflection', 'point')),
    (normalize,        ('bireflection',)),
    (triple_sandwich,  ('bireflection', 'point')),
    (chain,            ('vector',) * 5),
    (barycentric,      ('point', 'point', 'point', 'point')),
]

VGA_EXPRS = [
    (project,          ('vector', 'bivector')),
    (compose_and_apply,('bireflection', 'bireflection', 'vector')),
    (apply_twice,      ('bireflection', 'bireflection', 'vector')),
    (normalize,        ('bireflection',)),
    (triple_sandwich,  ('bireflection', 'vector')),
    (chain,            ('vector',) * 5),
]


def bench_expressions(rows, alg_name, factory):
    probe = factory()
    cases = PGA_EXPRS if hasattr(probe, 'point') else VGA_EXPRS
    print(f"\n  --- {alg_name} added operator expressions ---", flush=True)
    for fn, ctors in cases:
        for variant, spec in EXPR_VARIANTS:
            if not all(hasattr(probe, c) for c in ctors):
                continue
            state = {}

            def run_once(variant=variant, spec=spec):
                alg = factory(**({'cse': False} if spec.get('cse') is False else {}))
                mvs = [numeric(alg, c) for c in ctors]
                if spec.get('plain'):
                    # No decorator at all: the expression is re-interpreted on every call,
                    # and only the individual built-ins underneath it are cached.
                    added_op, call = None, (lambda: fn(*mvs))
                else:
                    added_op = alg.add_operator(symbolic=spec['symbolic'], name=f'{fn.__name__}_{variant}')(fn)
                    call = (lambda: added_op(*mvs))
                t0 = time.perf_counter()
                call()
                dt = time.perf_counter() - t0
                state['added_op'], state['mvs'], state['call'] = added_op, mvs, call
                return dt

            try:
                cold, cold_med, cold_n = time_cold(run_once)
            except Exception as e:
                rows.append(dict(section='expression', algebra=alg_name, d=probe.d,
                                 operation=fn.__name__, variant=variant,
                                 inputs='+'.join(ctors), out_keys='', muls='', divs='', adds='',
                                 cold_ms='', cold_med_ms='', cold_n='',
                                 hot_us='', hot_med_us='', hot_n='',
                                 note=f'{type(e).__name__}: {e}'))
                print(f"  {fn.__name__:<18} {variant:<15} FAILED {type(e).__name__}", flush=True)
                continue

            added_op, mvs = state['added_op'], state['mvs']
            hot, hot_med, hot_n = time_hot(state['call'])
            if added_op is None:                    # plain: no single fused function exists
                out_keys = muls = divs = adds = ''
            else:
                compiled = added_op[mvs]
                out_keys = len(compiled.keys_out)
                muls, divs, adds = op_counts(compiled)
            rows.append(dict(section='expression', algebra=alg_name, d=probe.d,
                             operation=fn.__name__, variant=variant,
                             inputs='+'.join(ctors), out_keys=out_keys,
                             muls=muls, divs=divs, adds=adds,
                             cold_ms=round(cold, 4), cold_med_ms=round(cold_med, 4), cold_n=cold_n,
                             hot_us=round(hot, 3), hot_med_us=round(hot_med, 3), hot_n=hot_n,
                             note=(fn.__doc__ or '').strip()))
            print(f"  {fn.__name__:<18} {variant:<15} {cold:9.3f} ms (n={cold_n})"
                  f"  {hot:8.2f} us", flush=True)


# ---------------------------------------------------------------------------
# Section 4: values_asarray, over the three TL;DR examples from docs/arrays.rst
# ---------------------------------------------------------------------------

# Coefficient dtype for the torch backends. float64 matches numpy's default so the
# comparison is like-for-like; on a real GPU you would usually reach for float32, which
# would flatter the GPU numbers considerably.
TORCH_DTYPE = 'float64'


def _array_backends():
    """
    The coefficient backends to compare, each with the ``values_asarray`` the array
    docs prescribe plus the handful of array primitives the examples need.

    A backend that cannot run here is still returned, carrying the reason, so the
    report says "no CUDA build" rather than quietly showing three categories.
    """
    backends = []
    try:
        import numpy as np
    except ImportError:
        return backends

    for name, kwargs, note in (
        ('numpy list', {}, 'coefficients as a list of ndarrays'),
        ('numpy asarray', {'values_asarray': np.asarray}, 'coefficients in one ndarray'),
    ):
        backends.append(dict(
            name=name, kwargs=kwargs, note=note, available=True, reason='',
            rand=(lambda *s: np.random.rand(*s)),
            randint=(lambda hi, s: np.random.randint(0, hi, size=s)),
            sumfn=np.sum, sync=None, seed=(lambda: np.random.seed(0))))

    try:
        import torch
    except ImportError:
        return backends

    def torch_asarray(values):
        # Straight from docs/arrays.rst: torch has no object dtype, so anything that is
        # not a stack of tensors (basis blades, symbolic coefficients) passes through.
        return (torch.stack(values)
                if values and all(isinstance(v, torch.Tensor) for v in values) else values)

    dtype = getattr(torch, TORCH_DTYPE)
    for dev in ('cpu', 'cuda'):
        ok = dev == 'cpu' or torch.cuda.is_available()
        # torch parallelises elementwise ops across cores while numpy does not, so the
        # thread count is part of the result and belongs in the record.
        threads = f', {torch.get_num_threads()} threads' if dev == 'cpu' else ''
        backends.append(dict(
            name=f'torch {dev}', kwargs={'values_asarray': torch_asarray},
            note=f'torch {TORCH_DTYPE} tensors on {dev}{threads}',
            available=ok,
            reason='' if ok else f'torch {torch.__version__} reports no CUDA device',
            rand=(lambda *s, dev=dev: torch.rand(*s, device=dev, dtype=dtype)),
            randint=(lambda hi, s, dev=dev: torch.randint(0, hi, s, device=dev)),
            sumfn=torch.sum,
            # CUDA kernels are async: without a sync the timer measures dispatch, not work.
            sync=((lambda: torch.cuda.synchronize()) if dev == 'cuda' else None),
            seed=(lambda: torch.manual_seed(0))))
    return backends


# Fixed so a backend that cannot run still labels its rows with the real case names,
# and lines up with the backends that did run instead of inventing extra categories.
ARRAY_CASE_NAMES = ('project_grid', 'mask_cloud', 'mesh_areas')


def _array_cases(be, alg, sizes):
    """The three vectorised patterns the array docs lead with, ready to call."""
    N, M, V, F = sizes
    rand, randint, sumfn = be['rand'], be['randint'], be['sumfn']

    # 1. Project N points onto M lines, broadcasting to an (N, M) grid.
    points = alg.point(rand(3, N))
    lines = alg.bivector(rand(6, M)).normalized()

    # 2. Mask a point cloud by distance to the origin.
    cloud = alg.point(rand(3, N))
    O = alg.point([0.0, 0.0, 0.0])

    # 3. Mesh: index V vertices by an (F, 3) face array, then take triangle areas.
    verts = rand(V, 3)
    faces = randint(V, (F, 3))
    v = (alg.blades.e0 + alg.evector(verts.T)).dual()

    def project():
        return points[:, None] @ lines[None, :]

    def mask():
        d = (cloud & O).norm()
        return cloud[d.e < 1]

    def mesh():
        facets = v[faces]
        planes = facets[..., 0] & facets[..., 1] & facets[..., 2]
        return (0.5 * planes.norm()).map(sumfn)

    return list(zip(ARRAY_CASE_NAMES,
                    [f'{N} points × {M} lines', f'{N} points', f'{V} verts, {F} faces'],
                    [project, mask, mesh]))


# small / medium / large is a single 4x ladder driven by one number, not three ad-hoc
# tuples: N points, N/4 lines to project them onto, 8N mesh vertices and 16N faces.
ARRAY_SCALES = [('small', 64), ('medium', 256), ('large', 1024)]


def _array_sizes(n):
    """(points, lines, vertices, faces) for scale ``n``."""
    return (n, max(1, n // 4), n * 8, n * 16)


def bench_arrays(rows, alg_name='3DPGA'):
    """
    Compare the array examples with and without ``values_asarray``.

    Without it a multivector holds a *list* of arrays, one per basis blade; with
    ``np.asarray`` the coefficients live in one array. Same maths either way, so this
    isolates what the constructor argument buys. It is a genuine trade rather than a
    free win: every multivector that gets built pays a stack-and-copy, which buys
    cheaper bulk indexing later -- so it is swept over three problem sizes.
    """
    backends = _array_backends()
    if not backends:
        print('  numpy not installed - skipping the array section', flush=True)
        return

    print(f"\n  --- {alg_name} array coefficient backends ---", flush=True)
    for size_name, scale in ARRAY_SCALES:
        sizes = _array_sizes(scale)
        for be in backends:
            for idx in range(3):
                base = dict(section='array', algebra=alg_name, d=4, variant=be['name'],
                            out_keys='', muls='', divs='', adds='',
                            cold_ms='', cold_med_ms='', cold_n='',
                            hot_us='', hot_med_us='', hot_n='')
                if not be['available']:
                    rows.append(dict(base, operation=f'{ARRAY_CASE_NAMES[idx]} ({size_name})',
                                     inputs=size_name, note=f"skipped: {be['reason']}"))
                    continue

                state = {}

                def run_once(idx=idx, be=be, sizes=sizes):
                    be['seed']()
                    alg = ALGEBRAS[alg_name](**be['kwargs'])
                    name, shape, call = _array_cases(be, alg, sizes)[idx]
                    sync = be['sync']
                    t0 = time.perf_counter()
                    call()
                    if sync: sync()
                    dt = time.perf_counter() - t0
                    state.update(name=name, shape=shape,
                                 call=(call if not sync else (lambda: (call(), sync()))))
                    return dt

                try:
                    cold, cold_med, cold_n = time_cold(run_once)
                    hot, hot_med, hot_n = time_hot(state['call'])
                except Exception as e:
                    print(f"  [{size_name} {idx}] {be['name']:<15} FAILED "
                          f"{type(e).__name__}: {e}", flush=True)
                    rows.append(dict(base, operation=f'case{idx} ({size_name})',
                                     inputs=size_name, note=f'{type(e).__name__}: {e}'))
                    continue

                rows.append(dict(base, operation=f"{state['name']} ({size_name})",
                                 inputs=state['shape'],
                                 cold_ms=round(cold, 4), cold_med_ms=round(cold_med, 4),
                                 cold_n=cold_n, hot_us=round(hot, 3),
                                 hot_med_us=round(hot_med, 3), hot_n=hot_n, note=be['note']))
                print(f"  {state['name'] + ' (' + size_name + ')':<24} {be['name']:<15}"
                      f" {cold:9.3f} ms (n={cold_n})  {hot:10.2f} us", flush=True)

    missing = [b['name'] for b in backends if not b['available']]
    if missing:
        print(f"  not measured: {', '.join(missing)}", flush=True)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=DEFAULT_OUT, help='CSV file to write')
    p.add_argument('--algebras', nargs='*', default=['2DPGA', '3DPGA', '3DVGA'],
                   choices=list(ALGEBRAS), metavar='NAME')
    p.add_argument('--sections', nargs='*',
                   default=['creation', 'operator', 'expression', 'array'],
                   choices=['creation', 'operator', 'expression', 'array'], metavar='SECTION')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip', nargs='*', default=[], metavar='OP:OPERANDS',
                   help='operator rows to leave out, e.g. outertan:multivector '
                        '(codegen for that one takes ~20 s in 3DPGA)')
    args = p.parse_args()

    random.seed(args.seed)
    # Several operators warn (non-versor sqrt, outerexp on mixed grades). Emitting the
    # very first warning in a process costs ~50 ms of linecache + stderr work, which has
    # nothing to do with codegen but lands squarely in whichever row triggers it.
    warnings.simplefilter('ignore')
    warmup()
    rows = []

    if 'creation' in args.sections:
        print("Algebra creation:", flush=True)
        bench_creation(rows)

    for name in args.algebras:
        factory = ALGEBRAS[name]
        if 'operator' in args.sections:
            bench_operators(rows, name, factory, skip=set(args.skip))
        if 'expression' in args.sections:
            bench_expressions(rows, name, factory)

    if 'array' in args.sections:
        bench_arrays(rows)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == '__main__':
    main()
