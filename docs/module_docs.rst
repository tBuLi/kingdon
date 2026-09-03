Module Documentation
====================

Algebra
-------

.. automodule:: kingdon.algebra
   :members:
   :undoc-members:

MultiVector
-----------

.. automodule:: kingdon.multivector
   :members:
   :undoc-members:

GA Operators
------------

The operators module holds the GA operations themselves, written as ordinary
functions on purely symbolic :class:`~kingdon.multivector.MultiVector` objects.
E.g. :func:`~kingdon.operators.gp` computes the geometric product between two
multivectors for the specific non-zero basis blades present in the input.
These functions are also what the multivector types use to define their
layouts, see :doc:`types`.

.. automodule:: kingdon.operators
   :members:
   :undoc-members:

Powers
------

.. automodule:: kingdon.powers
   :members:
   :undoc-members:

Codegen
-------

The codegen module turns the symbolic result of an operation into a python
function, applying Common Subexpression Elimination along the way.

.. automodule:: kingdon.codegen
   :members:
   :undoc-members:

Operator dicts
--------------

.. automodule:: kingdon.operator_dict
   :members:
   :undoc-members:

Matrix reps
-----------

.. automodule:: kingdon.matrixreps
   :members:
   :undoc-members:

Graph
-----

.. automodule:: kingdon.graph
   :members:
   :undoc-members:

Rational Polynomial
-------------------

.. automodule:: kingdon.polynomial
   :members:
   :undoc-members:

Taperecorder
------------

Used by `Algebra.jit` to generate code without symbolic optimization.

.. automodule:: kingdon.taperecorder
   :members:
   :undoc-members:

Einops backend
--------------

Import this module to register kingdon's multivectors with `einops`, see :doc:`arrays`.

.. automodule:: kingdon.einops_backend
   :members:
   :undoc-members:
