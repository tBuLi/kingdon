===========
Basic Usage
===========

The most important object in all of :code:`kingdon` is :code:`Algebra`::

    from kingdon import Algebra

An :code:`Algebra` has to be initiated with a number of positive, negative,
and null-dimensions, which are traditionally denoted by :code:`p`, :code:`q` and :code:`r`.
For example, in order to create a traditional 3D algebra we can initiate

.. code-block::

    >>> alg = Algebra(p=3, q=0, r=0)

The default setting for each of these variables is :code:`0`, and thus the more typical way to
create this algebra is :code:`alg = Algebra(3)`.

In order to create elements in this algebra, the :code:`Algebra` class provides a number of
convenience methods.
For example, a scalar is created by :code:`Algebra.scalar`,
a vector by :code:`Algebra.vector`, a bivector by :code:`Algebra.bivector`,
and a pseudoscalar by :code:`Algebra.pseudoscalar`.
We can also create arbitrary multivectors using :code:`Algebra.multivector`.

Let us create two vectors :code:`u` and :code:`v` and multiply them to create
a quaternion:

.. code-block::

    >>> u = alg.vector([1, 1])
    >>> v = alg.vector([0, 1])
    >>> R = u*v

Performance
-----------
Because :code:`kingdon` attempts to symbolically optimize expressions
using :code:`sympy` the first time they are called, the first call to any operation is always slow,
whereas subsequent calls have extremely good performance.
This is because :code:`kingdon` first leverages the sparseness of the input,
*and* subsequently uses symbolic optimization to eliminate any terms that are always zero
regardless of the input.
For example, the product :math:`\mathbf{e}_{1} \mathbf{e}_{12}` of the vector :math:`\mathbf{e}_1`
and the bivector :math:`\mathbf{e}_{12}` in :math:`\mathbb{R}_{2+p',q,r}` always returns
:math:`\mathbf{e}_2` for any :math:`p',q,r`. In :code:`kingdon`, it will also be equally fast to compute
in all of these algebras, regardless of the total dimension.
The same cannot be said for most other GA libraries.

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
