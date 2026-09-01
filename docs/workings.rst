===============
Developer Guide
===============
This chapter will explain how :code:`kingdon` works internally to make it easier to start contributing to kingdon.
(Under construction.)


Type system internals
---------------------

For how to *use* the type system, see :doc:`types`. What follows is how it works.

Binding layouts
~~~~~~~~~~~~~~~

The single representation of a type is its *layout*: a dict from blade key (the binary rep of a basis
blade) to either :code:`...` for a free component, or a number for a structural constant. Every
registered type is bound to a layout once, when the algebra is created, by
:code:`Algebra._bind_layout`, and the result is cached on
:code:`Algebra._type_layouts`.

A type that defines a :code:`layout` needs no more than a translation from canonical blade names to
(binary) keys. A type that defines an :code:`archetype` gets that archetype evaluated: the expression is run
with symbolic coefficients and the result is read off blade by blade. A coefficient that came out
numerical is a structural constant of the type, anything still symbolic is a free component. This is
also why archetypes must be written with :mod:`kingdon.operators` directly: they run during
:code:`Algebra.__post_init__`, before the operators have been registered on
:class:`~kingdon.multivector.MultiVector`.

Instances expose their bound layout as :attr:`~kingdon.multivector.MultiVector.type_layout`, and only
the free components are actually stored:

.. code-block::

    >>> pga = Algebra.fromname('3DPGA')
    >>> p = pga.point(name='p')
    >>> p.type_layout
    {14: Ellipsis, 13: Ellipsis, 11: Ellipsis, 7: 1.0}
    >>> p.keys()
    (14, 13, 11)

:math:`\mathbf{e}_{123}` (key :code:`7`) is in the layout but not in :code:`keys()`:
codegen substitutes it as a constant when compiling functions involving points,
but it never becomes an argument or a value in memory.

Resolving types
~~~~~~~~~~~~~~~

Codegen produces the layout of a result, and :func:`~kingdon.codegen.resolve_layout` decides which
type that is by comparing it against every entry of :code:`_type_layouts`. A registered type matches
when
- its fixed values agree with the result
- it does not fix anything the result leaves free
- it knows about every fixed blade the result carries.
Of the matching types the most specific wins:
first minimising free slots that coincide with fixed values of the result, then free slots outside
the result altogether, with ties broken by registration order. So in 3DPGA a :code:`Point` beats
:code:`Trivector` for a trivector with a :code:`1` on :math:`\mathbf{e}_{123}`, and
:class:`~kingdon.multivector.MultiVector` is the fall-back when nothing matches.

Since matching is purely structural, an operation is typed by what it produces:

.. code-block::

    >>> alg = Algebra(2)
    >>> type(alg.vector(name='u') * alg.vector(name='v'))
    <class 'kingdon.multivector.Bireflection'>

Blade orientation
~~~~~~~~~~~~~~~~~

A layout may order its blades freely, but the indivdual blades in the layout have to correspond
to (an even permutation of) a basis blade of the algebra.
Orientation is a property of the *basis*, set with the :code:`basis` argument of
:class:`~kingdon.algebra.Algebra`, not of an individual type.
