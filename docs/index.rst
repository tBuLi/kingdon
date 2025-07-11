Welcome to Kingdon's documentation!
======================================

.. image:: https://img.shields.io/pypi/v/kingdon.svg
        :target: https://pypi.python.org/pypi/kingdon

.. image:: https://readthedocs.org/projects/kingdon/badge/?version=latest
        :target: https://kingdon.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/tBuLi/kingdon/badge.svg?branch=master
        :target: https://coveralls.io/github/tBuLi/kingdon?branch=master


`✨ Try kingdon in your browser ✨ <https://tbuli.github.io/teahouse/>`_

Features
--------

Kingdon is a Geometric Algebra (GA) library which combines a Pythonic API with
symbolic simplification and just-in-time compilation to achieve high-performance in a single package.
It support both symbolic and numerical GA computations.
Moreover, :code:`kingdon` uses :code:`ganja.js` for visualization in notebooks,
making it an extremely well rounded GA package.

In bullet points:

- Symbolically optimized code generation.
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

   * - .. image:: _static/pga2d_distances_and_angles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_distances_and_angles.ipynb

       Land measurement 101
     - .. image:: _static/pga2d_inverse_kinematics.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_inverse_kinematics.ipynb

       Dimension agnostic IK
     - .. image:: _static/pga2d_project_and_reject.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_project_and_reject.ipynb

       2D projection and intersection
   * - .. image:: _static/pga3d_distances_and_angles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=3DPGA%2Fex_3dpga_distances_and_angles.ipynb

       Land measurement 420
     - .. image:: _static/pga2d_hypercube_on_string.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_hypercube_on_string.ipynb

       Best-seller: Tesseract on a string!
     - .. image:: _static/pga3d_points_and_lines.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=3DPGA%2Fex_3dpga_points_and_lines.ipynb

       3D projection and intersection
   * - .. image:: _static/exercise_spider6.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=exercises%2Fspider6.ipynb

       Build-A-Spider Workshop!
     - .. image:: _static/cga2d_points_and_circles.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DCGA%2Fex_2dcga_points_and_circles.ipynb

       Project and intersect, but round
     - .. image:: _static/pga2d_fivebar.png
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_fivebar.ipynb

       Fivebar mechanism
   * - .. image:: _static/csga2d_opns.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2F2DCSGA.ipynb

       2DCSGA!
     - .. image:: _static/mga3d_points_and_lines.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2FMotherAlgebra.ipynb

       Mother Algebra
     - .. image:: _static/ccga3d_points_quadrics.jpg
          :target: https://tbuli.github.io/teahouse/lab/index.html?path=OPNS%2F3DCCGA.ipynb

       3DCCGA


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   usage
   workings
   examples/index
   contributing
   authors
   module_docs
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
