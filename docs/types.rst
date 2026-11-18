===========
Type system
===========

Starting from kingdon v3, kingdon has a type system. The idea is simple: most multivectors we
compute with are not arbitrary. A normalized point in 3DPGA always has :code:`1` on
:math:`\mathbf{e}_{123}`, a translation always has :code:`1` on the scalar. Kingdon does not store
those values, and does not multiply, add or carry them around either, which is part of where the
shorter code in :ref:`the CSE table <usage:Performance Tips>` comes from.

Every algebra comes with a sensible set of :ref:`types <usage:Multivector Types>` already. To add your
own, pass them to :class:`~kingdon.algebra.Algebra` as :code:`extra_types` (added to the standard
list) or :code:`types` (replaces it). Each type gets a lowercase constructor on the algebra, and
there are three ways to define one.

A layout dict
~~~~~~~~~~~~~

A type is fundamentally described by its *layout*: a mapping from basis blade to either :code:`...` for a free
component, or a number for a value that is fixed by the type. The quickest way is to
hand one over directly:

.. code-block::

    >>> alg = Algebra(3, 0, 1, extra_types=[
    ...     {'name': 'MyPoint', 'layout': {'e023': ..., 'e013': ..., 'e012': ..., 'e123': 1}}])
    >>> alg.mypoint(name='p')
    p023 𝐞₀₂₃ + p013 𝐞₀₁₃ + p012 𝐞₀₁₂ + 1 𝐞₁₂₃

Only three values are stored here; the :math:`\mathbf{e}_{123}` comes with the type.

A layout dict on a class
~~~~~~~~~~~~~~~~~~~~~~~~

Give the layout as a class attribute instead when the type needs methods of its own. A layout may
list the blades in whatever order the type likes, the basis blades have to agree with the basis of
the algebra. E.g. for quaternions on :math:`\mathbf{e}_{23}, \mathbf{e}_{31}, \mathbf{e}_{12}`:

.. code-block::

    >>> class Quat(MultiVector):
    ...     layout = {'e': ..., 'e23': ..., 'e31': ..., 'e12': ...}
    >>> alg = Algebra(3, basis=["e", "e1", "e2", "e3", "e12", "e31", "e23", "e123"],
    ...               types=[Quat])
    >>> q = alg.quat([1., 2., 3., 4.])
    >>> q.e31
    3.0

The benefit of defining the class directly is that you can also add custom methods and attributes
while the kingdon type-system ensures that your types will be used whenever possible.
For example, the product of two :code:`Quat` will be a :code:`Quat`, not a generic multivector.

A layout classmethod
~~~~~~~~~~~~~~~~~~~~

A layout dict is tied to one algebra. To avoid having to define a type for every individual algebra you might use it in,
give your type a :code:`layout` classmethod instead: a GA expression that builds a fully symbolic example of
the type as a function of the algebra. Kingdon reads the layout off of that example.

.. code-block::

    import kingdon.operators as ops

    class Direction(MultiVector):
        @classmethod
        def layout(cls, algebra, name):
            return ops.polarity(Vector.layout(algebra, name))

    class Motor(MultiVector):
        @classmethod
        def layout(cls, algebra, name):
            return ops.gp(Bireflection.layout(algebra, f'{name}_1'),
                          Bireflection.layout(algebra, f'{name}_2'))

    alg = Algebra(3, 0, 1, extra_types=[Motor])
    m = alg.motor(name='m')

As shown, types can be built out of each other. Use the functions from :mod:`kingdon.operators`
rather than the usual :code:`p * ~q`, because these layouts are evaluated while the algebra is still
being created and so this syntactic sugar does not exist yet at that point.

Registering a type is all it takes for kingdon to find it, since results are matched to types
structurally. If nothing matches, you get a plain :class:`~kingdon.multivector.MultiVector`, so an
unregistered layout is never an error. See :doc:`workings` for how the matching works.

Types made of several multivectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some objects consist of several multivectors of the same type: a line segment is two points, a polyline any
number of them. Declare such a type by subscripting the type it is made of, with :code:`None` for a number that
is not fixed:

.. code-block::

    >>> class LineSegment(Point[2]): pass
    >>> class PolyLine(Point[None]): pass
    >>> alg = Algebra(3, 0, 1, extra_types=[Point, LineSegment, PolyLine])
    >>> segment = alg.linesegment(np.array([[0., 2.], [0., 0.], [0., 0.]]))

The values of a line segment have an extra dimension of length two right after the blades, one entry per point.
That dimension belongs to the type, so a single line segment has shape :code:`()`, and a batch of ten has shape
:code:`(10,)`. Cast to :code:`Point` to get at the points themselves:

.. code-block::

    >>> segment.shape
    LineSegment[()]
    >>> start, end = segment.asmvtype(Point)
    >>> end
    2.0 𝐞₀₁₂ + 1.0 𝐞₁₂₃

Operators treat the points alike and keep the type, so a transformed line segment is a line segment again:

.. code-block::

    >>> motor = alg.evenmv(np.random.rand(8))
    >>> type(motor >> segment)
    <class '__main__.LineSegment'>

Operator overloads
~~~~~~~~~~~~~~~~~~

Treating the points alike is not always what you want. Projecting a line segment onto a plane can project both
points, but the projection of a point onto a line segment should stay on the segment. For cases like that,
register a plain Python function on the operator. The operator then calls it as-is, instead of its generated
code, whenever the inputs match the type hints of its parameters:

.. code-block::

    class LineSegment(Point[2]):
        @property
        def points(self):
            return self.asmvtype(Point)

    alg = Algebra(3, 0, 1, extra_types=[Point, LineSegment])

    @alg.proj.register
    def _(x: LineSegment, y: MultiVector):
        return (x.points @ y).asmvtype(LineSegment)

Now :code:`segment @ plane` calls this function.

Matching is exact: an overload is called for inputs of exactly the types in its type hints. A subclass does not
inherit an overload, so a :code:`class MySegment(LineSegment)` needs overloads of its own, and a type hint
:code:`Point` does not match a :code:`LineSegment`. The one exception is :class:`~kingdon.multivector.MultiVector`,
which is also what a parameter without a type hint takes: it matches any multivector that has no overload of its own
type. When the exact types have no overload, the operator tries :code:`MultiVector` in place of one multivector
input, then of two, and so on, and the first overload it finds wins. Python numbers and other inputs that are not
multivectors never fall back to :code:`MultiVector`. If two overloads are found at the same step, as for
:code:`segment * segment` with overloads for :code:`(LineSegment, MultiVector)` and :code:`(MultiVector, LineSegment)`,
the operator raises a :code:`TypeError`: register an overload for the exact combination to settle it.

Register all overloads of an operator before you first use it: the operator remembers which overload, if any,
it chose for each combination of input types, and does not look again.

At least one type hint has to be a type made of several multivectors, such as :code:`LineSegment`, or a class
that is not a multivector at all, such as a mesh class of your own. The standard behaviour of the plain types is
not for overloading, so registering :code:`def _(x: Point, y: MultiVector)` raises a :code:`TypeError`.
Plain types can appear next to such a type hint, as in :code:`def _(x: Point, y: LineSegment)`.

Overloads on coefficient types, such as :code:`float` or :code:`np.ndarray`, are allowed, but whether they make
sense is up to you. An input that is callable but not a multivector is called before the operator looks for an
overload, so a callable class of your own cannot be used as a type hint.
