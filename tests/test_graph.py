from kingdon import Algebra
import numpy as np
import pytest

@pytest.mark.parametrize('alg', [Algebra(2, 0, 1), Algebra.fromname('2DPGA')])
def test_widget(alg):
    x = alg.vector([1, 1, 1]).dual()
    y = lambda: alg.vector([1, 1, 1]).dual()
    z = alg.vector([1, 1, 1])
    wvals = np.ones((3, 5))
    w = alg.vector(wvals).dual()
    wcoeffs = np.array(w.values())  # Not wvals: dualizing flips signs and reorders blades.
    p = alg.vector([1, 2, 1]).dual()
    func = lambda: x & y
    args = (0xD0FFE1, x, 0x00AA88, y, func, z, w, p)
    if alg.basis:  # TODO: remove this clause when #141 has been solved.
        with pytest.raises(ValueError):
            g = alg.graph(*args)
        return
    g = alg.graph(*args)
    # Only point 1 and 11 are draggable because they are the only direct PGA trivectors.
    assert g.draggable_points_idxs == [1, 11]
    assert g.draggable_points == [[
        {'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'},
        {'keys': p.keys(), 'mv': p.values(), 'type': 'bivector'},
    ]]
    assert all(isinstance(s, int) for s in g.signature)
    subjects = [
        0xD0FFE1,
        {'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'},
        0x00AA88,
        {'keys': y().keys(), 'mv': y().values(), 'type': 'bivector'},
        {'keys': func().keys(), 'mv': func().values(), 'type': 'vector'},
        {'keys': z.keys(), 'mv': z.values(), 'type': 'vector'},
        {'keys': w.keys(), 'mv': {'dtype': wcoeffs.dtype.str, 'shape': wcoeffs.shape, 'buffer': wcoeffs.tobytes()}, 'type': 'bivector'},
        {'keys': p.keys(), 'mv': p.values(), 'type': 'bivector'},
    ]
    assert g.subjects == subjects

    # Test if graph has the right basis, signature, and default style.
    assert g.basis == [b if b != 'e' else '1' for b in alg.canon2bin]
    assert g.signature == alg.signature
    assert g.options['style'] == {
        'width': 'min( 100%, 1024px )',
        'height': 'auto',
        'aspectRatio': '16 / 6',
        'background': 'white',
        'marginLeft': 'calc( (100% - min( 100%, 1024px )) / 2 )',
    }

    # Simulte dragging a point and see if the point updates. Ganja supplies a full multivector.
    x_prime = alg.vector([1, 1.01, 1]).dual().asfullmv()
    g.draggable_points = [{'keys': x_prime.keys(), 'mv': x_prime.values()}]
    assert all(getattr(x_prime, alg.bin2canon[k]) == v for k, v in x.items())

def test_up_function():
    """ Issue 93 implements the up function in graph, which enables OPNS rendering for exotic algebras like 2D CSGA. """
    import sympy as sp
    from sympy.printing.glsl import GLSLPrinter

    alg = Algebra(5, 3)
    e1, e2 = [alg.vector({key: 1}) for key in ['e1', 'e2']]
    p1, p2, p3 = [alg.vector({key: 1}) for key in ['e3', 'e4', 'e5']]
    n1, n2, n3 = [alg.vector({key: 1}) for key in ['e6', 'e7', 'e8']]

    # infinity (i) and origin (o) : plus (p), minus (m), times (t).
    ip, im, it = [n1 - p1, n2 - p2, n3 - p3]
    op, om, ot = alg.scalar(e=0.5) * [n1 + p1, n2 + p2, n3 + p3]

    # The 'up' (C) function that takes a Euclidean point and casts it into R5,3
    def up(x, y):
        return op + x * e1 + y * e2 + 0.5 * (x * x + y * y) * ip + 0.5 * (x * x - y * y) * im + x * y * it

    # The up function should be converted to a list of strings with valid GLSL syntax:
    up_mv = up(sp.Symbol('x'), sp.Symbol('y'))
    up_glsl = up_mv.map(GLSLPrinter().doprint)

    # Lets see what the graph object does.
    g = alg.graph(lambda: [], animate=0, up=up)
    assert g.options['up'] == up_glsl.values()

def test_update_125():
    alg = Algebra(2, 0, 1)
    x = alg.vector([1, 1, 1]).dual()
    y = lambda: alg.vector([1, 1, 1]).dual()
    Y = y()
    subjects = [0xD0FFE1, x, y]
    def graph_func(): return subjects
    g = alg.graph(graph_func)
    assert g.raw_subjects == [graph_func]
    assert g.pre_subjects == list(subjects)
    assert g.subjects == [
        0xD0FFE1,
        {'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'},
        {'keys': Y.keys(), 'mv': Y.values(), 'type': 'bivector'}
    ]
    # Only point 1 is draggable because it is the only PGA pseudovector.
    assert g.draggable_points_idxs == [1]
    assert g.draggable_points == [
        [{'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'}],
    ]

    z = alg.vector([1, 0.3, 1.2]).dual()
    new_subjects = (x, 0x00AA88, y, z)
    def new_graph_func(): return new_subjects
    options = {**g.options, 'scale': 4}

    # Now update the graph with new subjects and change the options.
    g.update(new_graph_func, **options)
    assert g.options == options
    assert g.raw_subjects == [new_graph_func]
    assert g.pre_subjects == list(new_subjects)
    assert g.subjects == [
        {'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'},
        0x00AA88,
        {'keys': Y.keys(), 'mv': Y.values(), 'type': 'bivector'},
        {'keys': z.keys(), 'mv': z.values(), 'type': 'bivector'}
    ]
    # Only point 1 is draggable because it is the only PGA pseudovector.
    assert g.draggable_points_idxs == [0, 3]
    assert g.draggable_points == [[
        {'keys': x.keys(), 'mv': x.values(), 'type': 'bivector'},
        {'keys': z.keys(), 'mv': z.values(), 'type': 'bivector'},
    ]]

# Only the structural constants of a layout are sent to js; free components are ignored.
@pytest.mark.parametrize('alg, types', [
    (Algebra(2, 0, 1),
     {'bireflection': {},
      'bivector': {},
      'scalar': {},
      'trivector': {},
      'vector': {}}
     ),
    (Algebra.fromname('2DPGA'),
     {'bireflection': {},
      'bivector': {},
      'direction': {},
      'evector': {},
      'point': {3: 1.0},
      'scalar': {},
      'translation': {0: 1.0},
      'trivector': {},
      'upoint': {4: 1.0},
      'vector': {}}
     )
])
def test_graph_types(alg, types):
    """test if types are correctly communicated to the GraphWidget."""
    globals().update(alg.blades.grade(1))

    if alg.basis:  # TODO: remove this clause when #141 has been solved.
        with pytest.raises(ValueError):
            g = alg.graph()
        return
    g = alg.graph()
    assert g.types == types
