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

Kingdon is an algebra for Geometric Algebra which aims to have a pythonic
and high-performance API for GA computation.
It was designed to work with numba to achieve high-performance and
broadcasts across numpy arrays so transformations can be applied easily to
e.g. point-clouds and meshes.

- Symbolically optimized.
- Numba enabled.
- Automatic broadcasting.

Code Example
------------
In order to demonstrate the power of Kingdon, let us first consider the common usecase of the
commutator product between a bivector and vector.


Symbolic usage:
.. code-block:: python
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector('b')
    >>> v = alg.vector('v')
    >>> b.cp(v)
    (-b12*v1 + b23*v3) * e2 + (b12*v2 + b13*v3) * e1 + (-b13*v1 - b23*v2) * e3 + (-b14*v1 - b24*v2 - b34*v3) * e4


Numerical:
.. code-block:: python
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector({'e12': 2})
    >>> v = alg.vector({'e1': 3})
    >>> b.cp(v)
    (-6) * e2


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
