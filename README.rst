=======
Kingdon
=======


.. image:: https://img.shields.io/pypi/v/kingdon.svg
        :target: https://pypi.python.org/pypi/kingdon

.. image:: https://readthedocs.org/projects/kingdon/badge/?version=latest
        :target: https://kingdon.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/tBuLi/kingdon/badge.svg?branch=master
        :target: https://coveralls.io/github/tBuLi/kingdon?branch=master



Pythonic Geometric Algebra Package


* Free software: MIT license
* Documentation: https://kingdon.readthedocs.io.

`✨ Try kingdon in your browser ✨ <https://tbuli.github.io/teahouse/>`_

Cite as::

    @misc{roelfs2025willingkingdoncliffordalgebra,
          title={The Willing Kingdon Clifford Algebra Library},
          author={Martin Roelfs},
          year={2025},
          eprint={2503.10451},
          archivePrefix={arXiv},
          primaryClass={cs.MS},
          url={https://arxiv.org/abs/2503.10451},
    }


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

Teahouse Menu
=============
If you are thirsty for some examples, please visit the `teahouse <https://tbuli.github.io/teahouse/>`_.
A small selection of our items:

.. list-table::
   :widths: 33 33 33
   :class: borderless

   * - .. image:: docs/_static/pga2d_distances_and_angles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_distances_and_angles.ipynb

       Land measurement 101
     - .. image:: docs/_static/pga2d_inverse_kinematics.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_inverse_kinematics.ipynb

       Dimension agnostic IK
     - .. image:: docs/_static/pga2d_project_and_reject.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_project_and_reject.ipynb

       2D projection and intersection
   * - .. image:: docs/_static/pga3d_distances_and_angles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=3DPGA%2Fex_3dpga_distances_and_angles.ipynb

       Land measurement 420
     - .. image:: docs/_static/pga2d_hypercube_on_string.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_hypercube_on_string.ipynb

       Best-seller: Tesseract on a string!
     - .. image:: docs/_static/pga3d_points_and_lines.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=3DPGA%2Fex_3dpga_points_and_lines.ipynb

       3D projection and intersection
   * - .. image:: docs/_static/exercise_spider6.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=exercises%2Fspider6.ipynb

       Build-A-Spider Workshop!
     - .. image:: docs/_static/cga2d_points_and_circles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DCGA%2Fex_2dcga_points_and_circles.ipynb

       Project and intersect, but round
     - .. image:: docs/_static/pga2d_fivebar.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_fivebar.ipynb

       Fivebar mechanism
   * - .. image:: docs/_static/csga2d_opns.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2F2DCSGA.ipynb

       2DCSGA!
     - .. image:: docs/_static/mga3d_points_and_lines.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2FMotherAlgebra.ipynb

       Mother Algebra
     - .. image:: docs/_static/ccga3d_points_quadrics.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2F3DCCGA.ipynb

       3DCCGA



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
    >>> b * v
    -6 𝐞₂

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
    b01 𝐞₀₁ + b02 𝐞₀₂ + b03 𝐞₀₃ + b12 𝐞₁₂ + b13 𝐞₁₃ + b23 𝐞₂₃
    >>> v = alg.vector(name='v')
    >>> v
    v0 𝐞₀ + v1 𝐞₁ + v2 𝐞₂ + v3 𝐞₃
    >>> b.cp(v)
    (b01*v1 + b02*v2 + b03*v3) 𝐞₀ + (b12*v2 + b13*v3) 𝐞₁ + (-b12*v1 + b23*v3) 𝐞₂ + (-b13*v1 - b23*v2) 𝐞₃

It is also possible to define some coefficients to be symbolic by inputting a string, while others can be numeric:

.. code-block:: python

    >>> from kingdon import Algebra, symbols
    >>> alg = Algebra(3, 0, 1)
    >>> b = alg.bivector(e12='b12', e03=3)
    >>> b
    3 𝐞₀₃ + b12 𝐞₁₂
    >>> v = alg.vector(e1=1, e3=1)
    >>> v
    1 𝐞₁ + 1 𝐞₃
    >>> w = b.cp(v)
    >>> w
    3 𝐞₀ + (-b12) 𝐞₂


A :code:`kingdon` MultiVector with symbols is callable. So in order to evaluate :code:`w` from the previous example,
for a specific value of :code:`b12`, simply call :code:`w`:

.. code-block:: python

    >>> w(b12=10)
    3 𝐞₀ + -10 𝐞₂


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
     -  $ab$
     - :code:`a*b`
     - :code:`a.gp(b)`
   * - Inner
     - $a \\cdot b$
     - :code:`a|b`
     - :code:`a.ip(b)`
   * - Scalar product
     - $\\langle a \\cdot b \\rangle_0$
     -
     - :code:`a.sp(b)`
   * - Left-contraction
     - $a \\rfloor b$
     -
     - :code:`a.lc(b)`
   * - Right-contraction
     - $a \\lfloor b$
     -
     - :code:`a.rc(b)`
   * - Outer (Exterior)
     - $a \\wedge b$
     - :code:`a ^ b`
     - :code:`a.op(b)`
   * - Regressive
     - $a \\vee b$
     - :code:`a & b`
     - :code:`a.rp(b)`
   * - Conjugate :code:`b` by :code:`a`
     - $a b \\widetilde{a}$
     - :code:`a >> b`
     - :code:`a.sw(b)`
   * - Project :code:`a` onto :code:`b`
     - $(a \\cdot b) \\widetilde{b}$
     - :code:`a @ b`
     - :code:`a.proj(b)`
   * - Commutator of :code:`a` and :code:`b`
     - $a \\times b = \\tfrac{1}{2} [a, b]$
     -
     - :code:`a.cp(b)`
   * - Anti-commutator of :code:`a` and :code:`b`
     - $\\tfrac{1}{2} \\{a, b\\}$
     -
     - :code:`a.acp(b)`
   * - Sum of :code:`a` and :code:`b`
     - $a + b$
     - :code:`a + b`
     - :code:`a.add(b)`
   * - Difference of :code:`a` and :code:`b`
     - $a - b$
     - :code:`a - b`
     - :code:`a.sub(b)`
   * - "Divide" :code:`a` by :code:`b`
     - $a b^{-1}$
     - :code:`a / b`
     - :code:`a.div(b)`
   * - Inverse of :code:`a`
     - $a^{-1}$
     -
     - :code:`a.inv()`
   * - Reverse of :code:`a`
     - $\\widetilde{a}$
     - :code:`~a`
     - :code:`a.reverse()`
   * - Grade Involution of :code:`a`
     - $\\hat{a}$
     -
     - :code:`a.involute()`
   * - Clifford Conjugate of :code:`a`
     - $\\bar{a} = \\hat{\\widetilde{a}}$
     -
     - :code:`a.conjugate()`
   * - Squared norm of :code:`a`
     - $a \\widetilde{a}$
     -
     - :code:`a.normsq()`
   * - Norm of :code:`a`
     - $\\sqrt{a \\widetilde{a}}$
     -
     - :code:`a.norm()`
   * - Normalize :code:`a`
     - $a / \\sqrt{a \\widetilde{a}}$
     -
     - :code:`a.normalized()`
   * - Square root of :code:`a`
     - $\\sqrt{a}$
     -
     - :code:`a.sqrt()`
   * - Dual of :code:`a`
     - $a*$
     -
     - :code:`a.dual()`
   * - Undual of :code:`a`
     -
     -
     - :code:`a.undual()`
   * - Grade :code:`k` part of :code:`a`
     - $\\langle a \\rangle_k$
     -
     - :code:`a.grade(k)`

Credits
-------

This package was inspired by GAmphetamine.js.
