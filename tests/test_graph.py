from kingdon import Algebra


def test_widget():
    alg = Algebra(2, 0, 1)
    x = alg.vector([1, 1, 1]).dual()
    y = lambda: alg.vector([1, 1, 1]).dual()
    z = alg.vector([1, 1, 1])
    args = (0xD0FFE1, x, 0x00AA88, y, lambda: x & y, z)
    g = alg.graph(*args)
    assert g.draggable_points_idxs == [1]
    assert g.draggable_points == [[{'keys': x.keys(), 'mv': x.values()}]]
    assert all(type(s) == int for s in g.signature)

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
