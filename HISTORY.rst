=======
History
=======

0.1.0 (2023-08-12)
------------------

* First release on PyPI.

0.2.0 (2024-01-09)
------------------

* Multivectors now have `map` and `filter` methods to apply element-wise operations to the coefficients.
* Make matrix representations of expressions using `expr_as_matrix`.
* Bugfixes.

0.3.0 (2024-03-11)
------------------
* Much faster codegen by the introduction of a GAmphetamine.js inspired RationalPolynomial class, which now replaces
  SymPy for codegen. Particularly for inverses this is orders of magnitude faster.
* Performed a numbotomy: numba is no longer a dependency since it actually didn't add much in most cases.
  Instead the user can now provide the Algebra with any wrapper function, which is applied to the generated functions.
  This can be numba.njit, but also any other decorator.
