====
Kingdon
====


.. image:: https://img.shields.io/pypi/v/kingdon.svg
        :target: https://pypi.python.org/pypi/kingdon

.. image:: https://img.shields.io/travis/tbuli/kingdon.svg
        :target: https://travis-ci.com/tbuli/kingdon

.. image:: https://readthedocs.org/projects/kingdon/badge/?version=latest
        :target: https://kingdon.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Pythonic Geometric Algebra Package


* Free software: MIT license
* Documentation: https://kingdon.readthedocs.io.


Features
--------

Kingdon is a Geometric Algebra (GA) library which aims to have a pythonic
and high-performance API for GA computation.
It was designed to work with numba to achieve high-performance and
broadcasts across numpy arrays so transformations can be applied easily to
e.g. point-clouds and meshes.

- Symbolically optimized.
- Numba enabled.
- Automatic broadcasting.

Code Example
------------
In order to demonstrate the power of Kingdon, let us first consider the common use-case of the
commutator product between a bivector and vector.

In order to create an algebra, use `Algebra`. When calling `Algebra` we must provide the signature of the
algebra, in this case we shall go for 3DPGA, which is the algebra :math:`\mathbb{R}_{3,0,1}`.
In order to make elements of the algebra, `kingdon` provides the functions `Algebra.multivector`, `Algebra.vector`, `Algebra.bivector`, etc.
These accept a sequence of values as their primary argument.
For example:

.. code-block:: python

    >>> from kingdon import Algebra
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector({'e12': 2})
    >>> v = alg.vector({'e1': 3})
    >>> b.cp(v)
    (-6) * e2

This example shows that only the `e2`coefficient is calculated, despite the fact that there are
6 bivector and 4 vector coefficients in 3DPGA. But because `kingdon` performs symbolic optimization before
performing the computation, it knows that in this case only `e2`can be non-zero.

Symbolic usage:

.. code-block:: python

    >>> from kingdon import Algebra
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector(name='b')
    >>> v = alg.vector(name='v')
    >>> b.cp(v)
    (-b12*v1 + b23*v3) * e2 + (b12*v2 + b13*v3) * e1 + (-b13*v1 - b23*v2) * e3 + (-b14*v1 - b24*v2 - b34*v3) * e4



Operators
=========
.. list-table:: Title
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
   * - Outer (Exterior)
     - :math:`a \wedge b`
     - :code:`a ^ b`
     - :code:`a.op(b)`
   * - Regressive
     - :math:`a \vee b`
     - :code:`a & b`
     - :code:`a.rp(b)`
   * - Conjugate :code:`b` by :code:`a`
     - :math:`a b \widetilde{a}`
     - :code:`a >> b`
     - :code:`a.sp(b)`
   * - Commutator :code:`a` by :code:`b`
     - :math:`a \times b = \tfrac{1}{2} [a, b]`
     -
     - :code:`a.cp(b)`

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
