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
:code:`Algebra.multivector` and explicitly pass a :code:`name` argument.
For example, let us create two symbolic vectors :code:`u` and :code:`v` in this algebra:

.. code-block::

    >>> u = alg.multivector(name='u', grades=(1,))
    >>> v = alg.multivector(name='v', grades=(1,))
    >>> u
    (u1) * e1 + (u2) * e2
    >>> v
    (v1) * e1 + (v2) * e2

The return type of :code:`Algebra.multivector` is an instance of :class:`~kingdon.multivector.MultiVector`.

.. note::
    :code:`kingdon` offers convenience methods for common types of multivectors, such as the vectors above.
    For example, the vectors above can also be created using :code:`u = alg.vector(name='u')`.
    Moreover, a scalar is created by :meth:`~kingdon.algebra.Algebra.scalar`, a bivector by :meth:`~kingdon.algebra.Algebra.bivector`,
    a pseudoscalar by :meth:`~kingdon.algebra.Algebra.pseudoscalar`, and so on.
    However, all of these merely add the corresponding :code:`grades` argument to your input and
    then call :code:`alg.multivector`, so :code:`alg.multivector` is what we need to understand.

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

We see that *in the case of vectors* the product is equal to the sum of the inner and exterior,
but this is **not the definition of the product**.

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

    >>> x = alg.multivector(name='x', keys=(0b01, 0b11))
    >>> (x1) * e1 + (x12) * e12

This can be done either by providing a tuple of integers which indicate which basis-vectors should be present,
or by passing them as strings, i.e. :code:`keys=('e1', 'e12')` is equivalent to the example above.
Internally however, :code:`kingdon` uses the binary representation.

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
     - :code:`b.conj(a)`
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

Note that formaly conjugation is defined by :math:`ba b^{-1}` and
projection by :math:`(a \cdot b) b^{-1}`, but that both are implemented
using reversion instead of an inverse. This is because reversion is much faster to calculate,
and because in practice :math:`b` will often by either a rotor satisfying
:math:`b \widetilde{b} = 1` or a blade satisfying :math:`b^2 = b \cdot b`,
and thus the inverse is identical to the reverse (up to sign).

If you want to replace these operators by their proper definitions, that is as easy as setting

.. code-block::

    >>> alg.conj.codegen = lambda x, y: x * y / y
    >>> alg.proj.codegen = lambda x, y: (x | y) / y

However, this comes with a huge performance cost for the first evaluation,
when codegen is performed for the given input, which is why this isn't the default.

.. warning::
    The syntax above for overwriting the codegen function might still be
    subject to change in the future, and is not guaranteed. However, the ability
    to customize the behavior of various operators, is guaranteed.
    The reason this might still change is because we want to add the ability to
    register any expression for codegen, not just unary and binary operators.


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
of :class:`~kingdon.multivector.MultiVector`, strings, and hexadecimal numbers to indicate colors.
These strings can be simple labels, or valid SVG syntax.

.. note::
    Currently :code:`ganja.js` support is limited to :mod:`jupyter` notebooks,
    and only static graphs are supported. In native :code:`ganja.js` lambda functions
    are evaluated every frame; this feature is currently not supported.

Performance
-----------
Because :code:`kingdon` attempts to symbolically optimize expressions
using :mod:`sympy` the first time they are called, the first call to any operation is always slow,
whereas subsequent calls have extremely good performance.
This is because :code:`kingdon` first leverages the sparseness of the input,
*and* subsequently uses symbolic optimization to eliminate any terms that are always zero
regardless of the input.
For example, the product :math:`\mathbf{e}_{1} \mathbf{e}_{12}` of the vector :math:`\mathbf{e}_1`
and the bivector :math:`\mathbf{e}_{12}` in :math:`\mathbb{R}_{2+p',q,r}` always returns
:math:`\mathbf{e}_2` for any :math:`p',q,r`.
In :code:`kingdon`, it will also be equally fast to compute this product in all of these algebras,
regardless of the total dimension.

Because the precomputation can get expensive, :code:`kingdon` predefines all the popular algebras
of :math:`d = p+q+r < 6`.
For example, a precomputed version of 3DPGA can be imported as

.. code-block::

    from kingdon.ga301 import ga301

It is also possible to cache the results of your script to your disk, such that subsequent runs of
the same script will not need to recompute products, but can instead load them from disk. This is
as easy as using the :code:`savecache` contextmanager around your script:

.. code-block::

    from kingdon import Algebra, savecache

    alg = Algebra(6)
    with savecache(alg) as alg:
        u = alg.vector(name='u')

By default this will store in the same location as the script, for more options
see the :code:`savecache`.
