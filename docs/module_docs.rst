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

Codegen
-------

The codegen module generates python functions from operations
between/on purely symbolic :class:`~kingdon.multivector.MultiVector` objects.

As a general rule, these functions take in pure symbolic
:class:`~kingdon.multivector.MultiVector` objects and return a tuple of keys
present in the output, and a pure python function which represents the
respective operation.

E.g. :func:`~kingdon.codegen.codegen_gp` computes the geometric product
between two multivectors for the specific non-zero basis blades present in
the input.

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

The GraphWidget enables interactive ganja.js graphing from notebooks.
This module also contains colormaps for your convenience with graphing.

.. code-block::

    >>> from kingdon import colormaps
    >>> cm = colormaps['Set2']


.. automodule:: kingdon.graph
   :members:
   :undoc-members:

Rational Polynomial
-------------------

.. automodule:: kingdon.polynomial
   :members:
   :undoc-members:
