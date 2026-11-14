=======
Kingdon
=======

.. rst-class:: kd-tagline

Pythonic Geometric Algebra Package

Features
--------

Kingdon is a Geometric Algebra (GA) library which combines a Pythonic API with
symbolic simplification and just-in-time compilation to achieve high-performance in a single package.

.. rst-class:: kd-features

- Symbolically optimized code generation
- Leverages sparseness of input
- Symbolic and numerical GA computations
- `ganja.js <https://github.com/enkimute/ganja.js>`__ graphics in jupyter notebooks
- Coefficients over `numpy <https://numpy.org/>`__, `torch <https://pytorch.org/>`__, `sympy <https://www.sympy.org/>`__, or anything with :code:`+`, :code:`-`, :code:`*`
- Automatic broadcasting, e.g. over point-clouds
- Compatible with `einops <https://einops.rocks/>`__ via :code:`kingdon.einops_backend`
- Compatible with `numba <https://numba.pydata.org/>`__ and other JIT compilers

Code Example
------------

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

**And it's** *fast*\ **.** New GAmphetamine-style CSE, on by default for built-in operators and optional for custom operators using :meth:`@alg.add_operator(symbolic=True) <kingdon.algebra.Algebra.add_operator>`.
3DPGA, counted in muls/adds — naive → CSE:
``R >> p`` 72/30 → **21/18** · ``p @ P`` 21/15 → **6/6** · ``P >> p`` 33/13 →
**9/7**. That's hand-optimized level, automatically generated.

Teahouse Menu
-------------
If you are thirsty for some examples, please visit the `teahouse <https://tbuli.github.io/teahouse/>`_.
A small selection of our items:

.. include:: _gallery.rst.inc

Cite as
-------

.. code-block:: bibtex

    @misc{roelfs2025willingkingdoncliffordalgebra,
          title={The Willing Kingdon Clifford Algebra Library},
          author={Martin Roelfs},
          year={2025},
          eprint={2503.10451},
          archivePrefix={arXiv},
          primaryClass={cs.MS},
          url={https://arxiv.org/abs/2503.10451},
    }

.. toctree::
   :caption: Getting started
   :hidden:

   installation
   usage
   operators

.. toctree::
   :caption: User guide
   :hidden:

   arrays
   types
   examples/index

.. toctree::
   :caption: Reference
   :hidden:

   api/index
   history

.. toctree::
   :caption: Development
   :hidden:

   workings
   contributing
   authors
