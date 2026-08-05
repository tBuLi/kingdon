===============
Developer Guide
===============
This chapter will explain how :code:`kingdon` works internally to make it easier to start contributing to kingdon.
(Under construction.)


Type system
-----------

Starting from kingdon v3, kingdon has a type system.

The idea is simple: a lot of the multivectors we compute with are not arbitrary. A normalized point
in 3DPGA always has :code:`1` on :math:`\mathbf{e}_{123}`, a translation always has :code:`1` on the
scalar. If kingdon knows this, it does not have to multiply by those ones, add them, or carry them
around. That is where the shorter code in :ref:`the CSE table <Performance Tips>` comes from.

Archetypes
~~~~~~~~~~

A multivector type is a subclass of :class:`~kingdon.multivector.MultiVector` with an :code:`archetype`
classmethod. The archetype is just a GA expression, written with the functions from
:mod:`kingdon.operators`, that produces a fully symbolic example (archetype) of the type. For instance a
bireflection is the geometric product of two vectors, and a PGA direction is the polar dual of one::

    import kingon.operators as ops

    class Bireflection(MultiVector):
        @classmethod
        def archetype(cls, algebra, name):
            p = Vector.archetype(algebra, f'{name}_1')
            q = Vector.archetype(algebra, f'{name}_2')
            return ops.gp(p, ops.reverse(q))

    class Direction(MultiVector):
        @classmethod
        def archetype(cls, algebra, name):
            return ops.polarity(Vector.archetype(algebra, name))

We must use :mod:`kingdon.operators` directly becuase the :code:`archetype` classmethod is run
during innitiation of :code:`Algebra`, before the syntactic sugar like :code:`p * ~q` exists.
Types can build on each other, in the way shown in :code:`Direction` above.
The archetype classmethod allows the multivector types to be defined in a dimension agnostic way:
the bireflection definition above is valid in all algebras, whereas the direction definition is valid in all PGAs.

Layouts
~~~~~~~

When an algebra is created it evaluates the archetype of every registered type, and reads off the
resulting coefficients. Anything that came out as a number is a structural constant of the type;
anything symbolic is a free component. This mapping from blade key to either :code:`...` (free) or a
number (fixed) is called the *layout*:

.. code-block::

    >>> pga = Algebra.fromname('3DPGA')
    >>> p = pga.point(name='p')
    >>> p.layout
    {14: Ellipsis, 13: Ellipsis, 11: Ellipsis, 7: 1.0}
    >>> p.keys()
    (14, 13, 11)

Note that :math:`\mathbf{e}_{123}` (key :code:`7`) is in the layout but not in :code:`keys()`: it is
fixed by the type.

Resolving types
~~~~~~~~~~~~~~~

After an operation, kingdon has the layout of the result and needs to decide what type it is. This is
:func:`~kingdon.codegen.resolve_layout`, which compares the result against every registered layout. A
type matches if its fixed values agree with the result, it does not fix anything the result leaves
free, and it knows about every fixed blade the result carries. Of the matching types the most
specific one wins, so e.g. in 3D PGA a :code:`Point` beats :code:`Trivector` when the result was a trivector with a :code:`1` on
:math:`\mathbf{e}_{123}`.

This also explains a result that surprises people at first:

.. code-block::

    >>> alg = Algebra(2)
    >>> type(alg.vector(name='u') * alg.vector(name='v'))
    <class 'kingdon.multivector.Bireflection'>

The product of two vectors is a bireflection, so that is what you get back. Types are matched
structurally, so a type only has to be registered to be found.

Registering your own
~~~~~~~~~~~~~~~~~~~~

Pass your own types to :class:`~kingdon.algebra.Algebra` with :code:`extra_types` to add to the
standard list, or :code:`types` to replace it entirely::

    class Motor(MultiVector):
        @classmethod
        def archetype(cls, algebra, name):
            return ops.gp(Bireflection.archetype(algebra, f'{name}_1'),
                          Bireflection.archetype(algebra, f'{name}_2'))

    alg = Algebra(3, 0, 1, extra_types=[Motor])
    m = alg.motor(name='m')

Every registered type gets a lowercase constructor on the algebra, so :code:`Motor` becomes
:code:`alg.motor`. Order matters a little: when two types fit equally well, the one registered first
wins. And if nothing matches at all, you simply get a plain
:class:`~kingdon.multivector.MultiVector`: it is always the fall-back, so an unregistered shape is
never an error.
