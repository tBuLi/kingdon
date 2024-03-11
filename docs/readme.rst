=======
Kingdon
=======


.. image:: https://img.shields.io/pypi/v/kingdon.svg
        :target: https://pypi.python.org/pypi/kingdon

.. image:: https://img.shields.io/travis/tbuli/kingdon.svg
        :target: https://travis-ci.com/tbuli/kingdon

.. image:: https://readthedocs.org/projects/kingdon/badge/?version=latest
        :target: https://kingdon.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/tBuLi/kingdon/badge.svg?branch=master
        :target: https://coveralls.io/github/tBuLi/kingdon?branch=master



Pythonic Geometric Algebra Package


* Free software: MIT license
* Documentation: https://kingdon.readthedocs.io.


Features
--------

Kingdon is a Geometric Algebra (GA) library which combines a Pythonic API with
symbolic simplification and just-in-time compilation to achieve high-performance in a single package.
It support both symbolic and numerical GA computations.
Moreover, :code:`kingdon` uses :code:`ganja.js` for visualization in notebooks,
making it an extremely well rounded GA package.

In bullet points:

- Symbolically optimized.
- Leverage sparseness of input.
- :code:`ganja.js` enabled graphics in jupyter notebooks.
- Agnostic to the input types: work with GA's over :code:`numpy` arrays, :code:`PyTorch` tensors, :code:`sympy` expressions, etc. Any object that overloads addition, subtraction and multiplication makes for valid multivector coefficients in :code:`kingdon`.
- Automatic broadcasting, such that transformations can be applied to e.g. point-clouds.
- Compatible with :code:`numba` and other JIT compilers to speed-up numerical computations.

Code Example
------------
In order to demonstrate the power of :code:`Kingdon`, let us first consider the common use-case of the
commutator product between a bivector and vector.

In order to create an algebra, use :code:`Algebra`. When calling :code:`Algebra` we must provide the signature of the
algebra, in this case we shall go for 3DPGA, which is the algebra :math:`\mathbb{R}_{3,0,1}`.
There are a number of ways to make elements of the algebra. It can be convenient to work with the basis blades directly.
We can add them to the local namespace by calling :code:`locals().update(alg.blades)`:

.. code-block:: python

    >>> from kingdon import Algebra
    >>> alg = Algebra(3, 0, 1)
    >>> locals().update(alg.blades)
    >>> b = 2 * e12
    >>> v = 3 * e1
    >>> b.cp(v)
    (-6) * e2

This example shows that only the :code:`e2` coefficient is calculated, despite the fact that there are
6 bivector and 4 vector coefficients in 3DPGA. But by exploiting the sparseness of the input and by performing symbolic
optimization, :code:`kingdon` knows that in this case only :code:`e2` can be non-zero.

Symbolic usage
--------------
If only a name is provided for a multivector, :code:`kingdon` will automatically populate all
relevant fields with symbols. This allows us to easily perform symbolic computations.

.. code-block:: python

    >>> from kingdon import Algebra
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector(name='b')
    >>> b
    (b12) * e12 + (b13) * e13 + (b23) * e23 + (b14) * e14 + (b24) * e24 + (b34) * e34
    >>> v = alg.vector(name='v')
    >>> v
    (v1) * e1 + (v2) * e2 + (v3) * e3 + (v4) * e4
    >>> b.cp(v)
    (-b12*v1 + b23*v3) * e2 + (b12*v2 + b13*v3) * e1 + (-b13*v1 - b23*v2) * e3 + (-b14*v1 - b24*v2 - b34*v3) * e4

It is also possible to define some coefficients to be symbolic by inputting a string, while others can be numeric::

    >>> from kingdon import Algebra, symbols
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector(e12='b12', e34=3)
    >>> b
    (b12) * e12 + (3) * e34
    >>> v = alg.vector(e1=1, e3=1)
    >>> v
    (1) * e1 + (1) * e3
    >>> w = b.cp(v)
    >>> w
    (-b12) * e2 + (-3) * e4


A :code:`kingdon` MultiVector with symbols is callable. So in order to evaluate :code:`w` from the previous example,
for a specific value of :code:`b12`, simply call :code:`w`::

    >>> w(b12=10)
    (-10) * e2 + (-3) * e4


Overview of Operators
=====================
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

