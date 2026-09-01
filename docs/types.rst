===========
Type system
===========

Starting from kingdon v3, kingdon has a type system. The idea is simple: most multivectors we
compute with are not arbitrary. A normalized point in 3DPGA always has :code:`1` on
:math:`\mathbf{e}_{123}`, a translation always has :code:`1` on the scalar. Kingdon does not store
those values, and does not multiply, add or carry them around either, which is part of where the
shorter code in :ref:`the CSE table <Performance Tips>` comes from.

Every algebra comes with a sensible set of :ref:`types <Multivector Types>` already. To add your
own, pass them to :class:`~kingdon.algebra.Algebra` as :code:`extra_types` (added to the standard
list) or :code:`types` (replaces it). Each type gets a lowercase constructor on the algebra, and
there are three ways to define one.

A layout
~~~~~~~~

A type is fundamentally described by its *layout*: a mapping from basis blade to either :code:`...` for a free
component, or a number for a value that is fixed by the type. The quickest way is to
hand one over directly:

.. code-block::

    >>> alg = Algebra(3, 0, 1, extra_types=[
    ...     {'name': 'MyPoint', 'layout': {'e023': ..., 'e013': ..., 'e012': ..., 'e123': 1}}])
    >>> alg.mypoint(name='p')
    p023 𝐞₀₂₃ + p013 𝐞₀₁₃ + p012 𝐞₀₁₂ + 1 𝐞₁₂₃

Only three values are stored here; the :math:`\mathbf{e}_{123}` comes with the type.

A layout on a class
~~~~~~~~~~~~~~~~~~~

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

An archetype
~~~~~~~~~~~~

A layout is tied to one algebra. To avoid having to define a type for every individual algebra you might use it in,
give your type an :code:`archetype` classmethod instead: a GA expression that builds a fully symbolic example of
the type as a function of the algebra.

.. code-block::

    import kingdon.operators as ops

    class Direction(MultiVector):
        @classmethod
        def archetype(cls, algebra, name):
            return ops.polarity(Vector.archetype(algebra, name))

    class Motor(MultiVector):
        @classmethod
        def archetype(cls, algebra, name):
            return ops.gp(Bireflection.archetype(algebra, f'{name}_1'),
                          Bireflection.archetype(algebra, f'{name}_2'))

    alg = Algebra(3, 0, 1, extra_types=[Motor])
    m = alg.motor(name='m')

As shown, types can be built out of each other. Use the functions from :mod:`kingdon.operators`
rather than the usual :code:`p * ~q`, because archetypes are evaluated while the algebra is still
being created and so this syntactic sugar does not exist yet when archetypes are born.

Registering a type is all it takes for kingdon to find it, since results are matched to types
structurally. If nothing matches, you get a plain :class:`~kingdon.multivector.MultiVector`, so an
unregistered layout is never an error. See :doc:`workings` for how the matching works.
