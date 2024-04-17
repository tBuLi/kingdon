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

0.3.2 (2024-03-18)
------------------
* Fixed a high priority bug in the graph function.
* Fixed a bug that stopped multivectors from being callable.

1.0.0 (2024-04-17)
------------------
* Kingdon now has proper support for ganja.js animations and the graphs are interactive!
* Indexing a multivector will no longer access coefficients.
  The whole promise of GA is coordinate independence, so why would you need to access coefficients?
  Instead, slicing a multivector will pass on that information to the underlying datastructures
  (e.g. numpy array or pytorch tensor), and will return a new multivector.
  Moreover, you can use the new slicing syntax to set values as well.
  If you really still need access to the coefficients, there is always the getattr syntax or the .values() method.
