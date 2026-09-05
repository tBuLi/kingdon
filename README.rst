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

.. image:: https://img.shields.io/pypi/dm/kingdon
        :target: https://pypi.python.org/pypi/kingdon
        :alt: PyPI - Downloads




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
Moreover, :code:`kingdon` uses `ganja.js <https://github.com/enkimute/ganja.js>`__ for visualization in notebooks,
making it an extremely well rounded GA package.

In bullet points:

- Symbolically optimized code generation.
- Leverage sparseness of input.
- `ganja.js <https://github.com/enkimute/ganja.js>`__ enabled graphics in jupyter notebooks.
- Agnostic to the input types: work with GA's over `numpy <https://numpy.org/>`__ arrays, `torch <https://pytorch.org/>`__ tensors, `sympy <https://www.sympy.org/>`__ expressions, etc. Any object that overloads addition, subtraction and multiplication makes for valid multivector coefficients in :code:`kingdon`.
- Automatic broadcasting, such that transformations can be applied to e.g. point-clouds.
- Compatible with `einops <https://einops.rocks/>`__ if you :code:`import kingdon.einops_backend` before you do your einops magic.
- Compatible with `numba <https://numba.pydata.org/>`__ and other JIT compilers to speed-up numerical computations.

Array Syntax
============
Kingdon has great symbiosis with the python `array api <https://data-apis.org/array-api/latest/>`_, allowing you to construct multidimensional multivectors using NumPy, PyTorch, JAX, CuPy, Dask, and more. (The examples below use NumPy.)
A multivector over arrays is just a batch of geometry — and the shape tells you the type:
 
.. code-block:: python
 
   alg = Algebra.fromname("3DPGA")
   points = alg.point(np.random.rand(3, 5))                   # Point[(5,)]
   lines  = alg.bivector(np.random.rand(6, 3)).normalized()   # Bivector[(3,)]
 
``kingdon`` supports vectorized expressions in addition to for-loops:
 
.. code-block:: python
 
   points[:, None] @ lines[None, :]   # Point[(5, 3)]: every point projected onto every line

The ``a @ b = (a | b) / b`` projection operator projects every point onto every line in one simple expressions giving a ``Point[(5, 3)]``.
Therefore `kingdon` allows you to write high-level algorithms that focus purely on the
geometry, while delegating the looping to the array library of your choice.
**Masking** passes straight through to your coefficients, so numpy tricks just
work:
 
.. code-block:: python
 
   O = alg.blades.e0.dual()                     # origin
   nearby = points[(points & O).norm().e < 1]   # every point within the unit sphere
 
**Fancy indexing** means a whole mesh is loop-free:
 
.. code-block:: python
 
   v      = alg.point(vertices.T)                             # Point[(N,)]    the point cloud
   facets = v[faces]                                          # Point[(M, 3)]  a point per face corner
   planes = facets[..., 0] & facets[..., 1] & facets[..., 2]  # Vector[(M,)]   the face planes
   area   = 0.5 * reduce(planes.norm(), 'm -> ', 'sum').e     # total surface area
   volume = reduce(planes, 'm -> ', 'sum').e0 / 6             # signed volume of the mesh
 
Yes, the signed volume of the whole mesh is just the sum of the ``e0`` coefficients. 🤯

But what if you do not want to manipulate the blade dimension (and hence the geometry), but you want to manipulate the batch dimensions instead?
For that, you can directly use **einops on multivectors**, making it easy to write high-level operations
such as ``rearrange``, ``reduce``, ``repeat``, ``pack``/``unpack``, ``einsum``,
without any reference to a specific array package. 
Patterns only ever mention the batch dims (the blade axis is not
yours to play with), so e.g. a vector stays a vector:
 
.. code-block:: python
 
   import kingdon.einops_backend
   from einops import rearrange, reduce, repeat
 
   x = alg.vector(np.random.rand(4, 3, 4))   # Vector[(3, 4)]
   rearrange(x, 'a b -> b a')                # Vector[(4, 3)]
   reduce(x, 'a b -> a', 'mean')             # Vector[(3,)]
   repeat(x, 'a b -> a b c', c=5)            # Vector[(3, 4, 5)]
 
This works for any package supported by einops (NumPy, PyTorch, JAX, CuPy, and more).
 
``pack``/``unpack`` glue multivectors together along a wildcard axis, carrying
the **right dtype** and living on the **right device**.
 
.. code-block:: python
 
   a = alg.vector(np.ones([4, 3, 5]))        # Vector[(3, 5)]
   b = alg.vector(np.ones([4, 3, 7, 5]))     # Vector[(3, 7, 5)]
   packed, ps = pack([a, b], 'j * k')        # Vector[(3, 8, 5)]
   a2, b2 = unpack(packed, ps, 'j * k')      # Vector[(3, 5)], Vector[(3, 7, 5)]
 
And ``einsum`` contracts batch dimensions blade by blade, so multivectors and
plain arrays mix freely:
 
.. code-block:: python
 
   vec = alg.vector(np.random.randn(4, 10, 10))   # Vector[(10, 10)]
   w   = np.random.randn(10, 20)                  # just a numpy array
   einsum(vec, 'i i ->')                          # Vector[()]         the trace
   einsum(vec, w, 'i j, j k -> i k')              # Vector[(10, 20)]   batched matmul
 


**And it's** *fast*\ **.** New GAmphetamine-style CSE, on by default for built-in operators and optional for custom operators using `@alg.add_operator(symbolic=True) <https://kingdon.readthedocs.io/en/stable/module_docs.html#kingdon.algebra.Algebra.add_operator>`_. 
3DPGA, counted in muls/adds — naive → CSE:
``R >> p`` 72/30 → **21/18** · ``p @ P`` 21/15 → **6/6** · ``P >> p`` 33/13 →
**9/7**. That's hand-optimized level, automatically generated.

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
We can add them to the local namespace by calling :code:`globals().update(alg.blades)`:

.. code-block:: python

    >>> from kingdon import Algebra
    >>> alg = Algebra(3, 0, 1)
    >>> globals().update(alg.blades)
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
   * - Conjugate :code:`a` by :code:`b` with :math:`\widetilde{b}b = 1`
     - $\\left(-1\\right)^{\\text{grade}\\left(b\\right) \\text{grade}\\left(a\\right)} b a \\widetilde{b}$
     - :code:`b >> a`
     - :code:`b.sw(a)`
   * - Project :code:`a` onto :code:`b` with :math:`\widetilde{b}b = 1`
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

This package was inspired by `GAmphetamine.js <https://github.com/enkimute/GAmphetamine.js>`__.
