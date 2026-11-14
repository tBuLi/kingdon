=======
Kingdon
=======

.. rst-class:: kd-tagline

Pythonic Geometric Algebra Package

.. rst-class:: kd-links

:doc:`Get started <usage>` `✨ Try kingdon in your browser ✨ <https://tbuli.github.io/teahouse/>`_ `GitHub <https://github.com/tbuli/kingdon>`_

.. rst-class:: kd-badges

|pypi| |rtd| |coveralls|

.. |pypi| image:: https://img.shields.io/pypi/v/kingdon.svg
   :target: https://pypi.python.org/pypi/kingdon

.. |rtd| image:: https://readthedocs.org/projects/kingdon/badge/?version=latest
   :target: https://kingdon.readthedocs.io/en/latest/?version=latest
   :alt: Documentation Status

.. |coveralls| image:: https://coveralls.io/repos/github/tBuLi/kingdon/badge.svg?branch=master
   :target: https://coveralls.io/github/tBuLi/kingdon?branch=master

Features
--------

Kingdon is a Geometric Algebra (GA) library which combines a Pythonic API with
symbolic simplification and just-in-time compilation to achieve high-performance in a single package.
It support both symbolic and numerical GA computations.
Moreover, :code:`kingdon` uses `ganja.js <https://github.com/enkimute/ganja.js>`__ for visualization in notebooks,
making it an extremely well rounded GA package.

- Symbolically optimized code generation.
- Leverage sparseness of input.
- `ganja.js <https://github.com/enkimute/ganja.js>`__ enabled graphics in jupyter notebooks.
- Agnostic to the input types: work with GA's over `numpy <https://numpy.org/>`__ arrays, `torch <https://pytorch.org/>`__ tensors, `sympy <https://www.sympy.org/>`__ expressions, etc. Any object that overloads addition, subtraction and multiplication makes for valid multivector coefficients in :code:`kingdon`.
- Automatic broadcasting, such that transformations can be applied to e.g. point-clouds.
- Compatible with `einops <https://einops.rocks/>`__ if you :code:`import kingdon.einops_backend` before you do your einops magic.
- Compatible with `numba <https://numba.pydata.org/>`__ and other JIT compilers to speed-up numerical computations.

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

.. toctree::
   :caption: User guide
   :hidden:

   arrays
   types
   examples/index

.. toctree::
   :caption: Reference
   :hidden:

   module_docs
   history

.. toctree::
   :caption: Development
   :hidden:

   workings
   contributing
   authors
