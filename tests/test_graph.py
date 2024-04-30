from kingdon import Algebra


def test_widget():
    alg = Algebra(2, 0, 1)
    x = alg.vector([1, 1, 1]).dual()
    y = lambda: alg.vector([1, 1, 1]).dual()
    args = (0xD0FFE1, x, 0x00AA88, y, lambda: x & y)
    g = alg.graph(*args)
    assert g.draggable_points_idxs == [1]
    assert g.draggable_points == [[{'keys': x.keys(), 'mv': x.values()}]]
    assert all(type(s) == int for s in g.signature)

