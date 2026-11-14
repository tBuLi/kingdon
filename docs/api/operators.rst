kingdon.operators
=================

.. automodule:: kingdon.operators

.. currentmodule:: kingdon.operators

These functions are also what the multivector types use to define their
layouts, see :doc:`/types`.

Products
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   product
   gp
   sw
   cp
   acp
   ip
   lc
   rc
   sp
   proj
   op
   rp
   square

Sums
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   add
   sub
   neg

Involutions
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   involutions
   reverse
   involute
   conjugate
   grade

Norms and roots
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   normsq
   sqrt

Inverses and division
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   inv
   hitzer_inv
   shirokov_inv
   div
   Fraction

Duals
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   polarity
   unpolarity
   hodge
   unhodge

Outer functions
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   outerexp
   outersin
   outercos
   outertan

Helpers
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   is_zero
   dict_to_multivector
   scalar
