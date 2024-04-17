===========
Basic Usage
===========

The most important object in all of :code:`kingdon` is :code:`Algebra`::

    from kingdon import Algebra

An :code:`Algebra` has to be initiated with a number of positive, negative,
and null-dimensions, which are traditionally denoted by :code:`p`, :code:`q` and :code:`r`.
For example, in order to create a 2D Geometric Algebra we can initiate

.. code-block::

    >>> alg = Algebra(p=2, q=0, r=0)
    >>> alg = Algebra(2)  # Equivalent: default value for p, q, r is 0.

The basis blades of the algebra are available in the dictionary :code:`alg.blades`. This can be
added to :code:`locals()` in order to allow for easy access to all the basis blades, and allows
the initiation of multivectors using the basis-blades directly:

.. code-block::

    >>> locals().update(alg.blades)
    >>> x = 2 * e + 1 * e1 - 5 * e2 + 6 * e12

where :code:`e` is the identity element, i.e. :math:`e = 1`.
This way of creating multivectors is particularly useful when writing quick scripts
in an interactive environment.
Let's look at some more general ways of making multivectors, starting with symbolic
multivectors before we go on to numerical multivectors.

Symbolic Multivectors
---------------------

In order to create symbolical multivectors in an algebra, we can call
:class:`~kingdon.algebra.Algebra.multivector` and explicitly pass a :code:`name` argument.
For example, let us create two symbolic vectors :code:`u` and :code:`v` in this algebra:

.. code-block::

    >>> u = alg.multivector(name='u', grades=(1,))
    >>> v = alg.multivector(name='v', grades=(1,))
    >>> u
    (u1) * e1 + (u2) * e2
    >>> v
    (v1) * e1 + (v2) * e2

The return type of :meth:`~kingdon.algebra.Algebra.multivector` is an instance of :class:`~kingdon.multivector.MultiVector`.

.. note::
    :code:`kingdon` offers convenience methods for common types of multivectors, such as the vectors above.
    For example, the vectors above can also be created using :code:`u = alg.vector(name='u')`.
    Moreover, a scalar is created by :meth:`~kingdon.algebra.Algebra.scalar`,
    a bivector by :meth:`~kingdon.algebra.Algebra.bivector`,
    a pseudoscalar by :meth:`~kingdon.algebra.Algebra.pseudoscalar`, and so on.
    However, all of these merely add the corresponding :code:`grades` argument to your input and
    then call :class:`~kingdon.algebra.Algebra.multivector`, so :class:`~kingdon.algebra.Algebra.multivector` is what we need to understand.

:class:`~kingdon.multivector.MultiVector`'s support common math operators:

.. code-block::

    >>> u + v
    (u1 + v1) * e1 + (u2 + v2) * e2
    >>> u * v
    (u1*v1 + u2*v2) + (u1*v2 - u2*v1) * e12

We also have the inner and exterior "products":

.. code-block::

    >>> u | v
    (u1*v1 + u2*v2)
    >>> u ^ v
    (u1*v2 - u2*v1) * e12

We see that *in the case of vectors* the product is equal to the sum of the inner and exterior.

Since vectors in 2DVGA represent reflections in lines through the origin, we can reflect the
line :code:`v` in the line :code:`u` by using conjugation:

.. code-block::

    >>> u >> v
    (u1**2*v1 + 2*u1*u2*v2 - u2**2*v1) * e1 + (-u1**2*v2 + 2*u1*u2*v1 + u2**2*v2) * e2

we see that the result is again a vector, as it should be.

These examples should show that the symbolic multivectors of :code:`kingdon`
make it easy to do symbolic computations. Moreover, we can also use :mod:`sympy` expressions
as values for the multivector:

.. code-block::

    >>> from sympy import Symbol, sin, cos
    >>> t = Symbol('t')
    >>> x = cos(t) * e + sin(t) * e12
    >>> x.normsq()
    1

More control over basisvectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we do not just want to create a symbolic multivector of a certain grade,
but with specific blades, we can do so by providing the :code:`keys` argument.

.. code-block::

    >>> x = alg.multivector(name='x', keys=('e1', 'e12'))
    >>> (x1) * e1 + (x12) * e12

This can be done either by providing a tuple of strings which indicate which basis-vectors should be present,
or by passing them as integers, i.e. :code:`keys=(0b01, 0b11)` is equivalent to the example above.
Internally, :code:`kingdon` uses the binary representation.

Numerical Multivectors
----------------------
While :code:`kingdon` makes no assumptions about the data structures that are passed into a multivector
in order to support ducktyping and customization as much as possible, it was nonetheless designed to
work really well with :code:`numpy` arrays.

For example, to repeat some of the examples above with numerical values, we could do

.. code-block::

    >>> import numpy as np
    >>> uvals, vvals = np.random.random((2, 2))
    >>> u = alg.vector(uvals)
    >>> v = alg.vector(vvals)
    >>> u * v
    (0.1541) + (0.0886) * e12

A big performance bottleneck that we suffer from in Python, is that arrays over objects are very slow.
So while we could make a numpy array filled with :code:`~kingdon.multivector.MultiVector`'s, this would tank our performance.
:code:`kingdon` gets around this problem by instead accepting numpy arrays as input. So to make a collection of
3 lines, we do

.. code-block::

    >>> import numpy as np
    >>> uvals = np.random.random((2, 3))
    >>> u = alg.vector(uvals)
    >>> u
    ([0.82499172 0.71181276 0.98052928]) * e1 + ([0.53395072 0.07312351 0.42464341]) * e2

what is important here is that the first dimension of the array has to have the expected length: 2 for a vector.
All other dimensions are not used by :code:`kingdon`. Now we can reflect this multivector in the :code:`e1` line:

.. code-block::

    >>> v = alg.vector((1, 0))
    >>> v >> u
    ([0.82499172 0.71181276 0.98052928]) * e1 + ([-0.53395072 -0.07312351 -0.42464341]) * e2

Despite the different shapes, broadcasting is done correctly in the background thanks to the magic of numpy,
and with only minor performance penalties.

Operators
---------

Instances of :mod:`~kingdon.multivector.MultiVector` overload all common Geometric Algebra operators.
Below is an overview:

.. list-table:: Operators
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operation
     - Expression
     - Infix
     - Inline
   * - Geometric product
     - :math:`ab`
     - :code:`a*b`
     - :code:`a.gp(b)`
   * - Inner
     - :math:`a \cdot b`
     - :code:`a|b`
     - :code:`a.ip(b)`
   * - Scalar product
     - :math:`\langle a \cdot b \rangle_0`
     - :code:`(a|b).grade(0)`
     - :code:`a.sp(b)`
   * - Left-contraction
     - :math:`a \rfloor b`
     -
     - :code:`a.lc(b)`
   * - Right-contraction
     - :math:`a \lfloor b`
     -
     - :code:`a.rc(b)`
   * - Outer (Exterior)
     - :math:`a \wedge b`
     - :code:`a ^ b`
     - :code:`a.op(b)`
   * - Regressive
     - :math:`a \vee b`
     - :code:`a & b`
     - :code:`a.rp(b)`
   * - Conjugate :code:`a` by :code:`b`
     - :math:`b a \widetilde{b}`
     - :code:`b >> a`
     - :code:`b.sw(a)`
   * - Project :code:`a` onto :code:`b`
     - :math:`(a \cdot b) \widetilde{b}`
     - :code:`a @ b`
     - :code:`a.proj(b)`
   * - Commutator of :code:`a` and :code:`b`
     - :math:`a \times b = \tfrac{1}{2} [a, b]`
     -
     - :code:`a.cp(b)`
   * - Anti-commutator of :code:`a` and :code:`b`
     - :math:`\tfrac{1}{2} \{a, b\}`
     -
     - :code:`a.acp(b)`
   * - Sum of :code:`a` and :code:`b`
     - :math:`a + b`
     - :code:`a + b`
     - :code:`a.add(b)`
   * - Difference of :code:`a` and :code:`b`
     - :math:`a - b`
     - :code:`a - b`
     - :code:`a.sub(b)`
   * - Reverse of :code:`a`
     - :math:`\widetilde{a}`
     - :code:`~a`
     - :code:`a.reverse()`
   * - Squared norm of :code:`a`
     - :math:`a \widetilde{a}`
     -
     - :code:`a.normsq()`
   * - Norm of :code:`a`
     - :math:`\sqrt{a \widetilde{a}}`
     -
     - :code:`a.norm()`
   * - Normalize :code:`a`
     - :math:`a / \sqrt{a \widetilde{a}}`
     -
     - :code:`a.normalized()`
   * - Square root of :code:`a`
     - :math:`\sqrt{a}`
     -
     - :code:`a.sqrt()`
   * - Dual of :code:`a`
     - :math:`a*`
     -
     - :code:`a.dual()`
   * - Undual of :code:`a`
     -
     -
     - :code:`a.undual()`
   * - Grade :code:`k` part of :code:`a`
     - :math:`\langle a \rangle_k`
     -
     - :code:`a.grade(k)`


Note that formally conjugation is defined by :math:`ba b^{-1}` and
projection by :math:`(a \cdot b) b^{-1}`, but that both are implemented
using reversion instead of an inverse. This is because reversion is much faster to calculate,
and because in practice :math:`b` will often by either a rotor satisfying
:math:`b \widetilde{b} = 1` or a blade satisfying :math:`b^2 = b \cdot b`,
and thus the inverse is identical to the reverse (up to sign).

If you want to replace these operators by their proper definitions, you can use the register decorator to
overwrite the default operator (use at your own risk):


.. code-block::

    >>> @alg.register(name='sw')
    >>> def sw(x, y):
    >>>     return x * y / y
    >>> @alg.register(name='proj')
    >>> def proj(x, y):
    >>>     return (x | y) / y


Graphing using :code:`ganja.js`
-------------------------------

:code:`kingdon` supports the :code:`ganja.js` graphing syntax. For those already familiar with
:code:`ganja.js`, the API will feel very similar:

.. code-block::

    >>> alg.graph(0xff0000, u, "u", lineWidth=3)

The rules are simple: all positional arguments will be passed on to :code:`ganja.js` as
elements to graph, whereas keyword arguments are passed to :code:`ganja.js` as options.
Hence, the example above will graph the line :code:`u` with :code:`lineWidth = 3`,
and will attach the label "u" to it, and all of this will be red.
Identical to :code:`ganja.js`, valid inputs to :code:`alg.graph` are (lists of) instances
of :class:`~kingdon.multivector.MultiVector`, strings, and hexadecimal numbers to indicate colors,
or a function without arguments that returns these things.
The strings can be simple labels, or valid SVG syntax.

.. note::
    kingdon supports :code:`ganja.js`'s animation and interactivity in jupyter notebooks,
    `try kingdon in your browser <https://tbuli.github.io/teahouse/>`_ to give it a go!

Performance Tips
----------------
Because :code:`kingdon` attempts to symbolically optimize expressions
using :mod:`sympy` the first time they are called, the first call to any operation is comparatively slow,
whereas subsequent calls have very good performance.

There are however several things to be aware of to ensure good performance.

Graded
~~~~~~
The first time :code:`kingdon` is asked to perform an operation it hasn't seen before, it performs code generation
for that particular request. Because codegen is the most expensive step, it is beneficial to reduce the number of
times it is needed. An easy way to achieve this is to initiate the :class:`~kingdon.algebra.Algebra` with `graded=True`.
This enforces that :code:`kingdon` does not specialize codegen down to the individual basis blades, but rather only
per grade. This means there are far less combinations that have to be considered and generated.

Numba JIT
~~~~~~~~~
We can enable numba just-in-time compilation by initiating an :class:`~kingdon.algebra.Algebra` with `wrapper=numba.njit`.
This comes with a significant cost the first time any operator is called, but subsequent calls to the same operator are
significantly faster. It is worth mentioning that when dealing with :ref:`Numerical Multivectors` over numpy arrays,
the benefit of using `numba` actually reduces rapidly as the numpy arrays become larger, since then most of the time
is spend in numpy routines anyway.

Register Expressions
~~~~~~~~~~~~~~~~~~~~
To make it easy to optimize larger expressions, :code:`kingdon` offers the :func:`~kingdon.algebra.Algebra.register`
decorator.

.. code-block::

    >>> alg = Algebra(3, 0, 1)
    >>>
    >>> @alg.register
    >>> def myfunc(u, v):
    >>>      return u * (u + v)
    >>>
    >>> x = alg.vector(np.random.random(4))
    >>> y = alg.vector(np.random.random(4))
    >>> myfunc(x, y)

Calling the decorated :code:`myfunc` has the benefit that all the numerical computation is done in one single call,
instead of doing each binary operation individually. This has the benefit that all the (expensive) python boilerplate
code is called only once.
