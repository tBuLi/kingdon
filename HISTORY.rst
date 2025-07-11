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

1.0.5 (2024-06-26)
------------------
* Blades by grade syntax: alg.blades.grade(2).
* Fixed "define" error in ganja.js integration, kingdon now works with reveal.js voila template.

1.0.6 (2024-07-10)
------------------
Bugfixes to ganja.js integration:
* Make sure camera is an object before checking for 'mv' key.
* Improved draggable points for PGA.

1.1.0 (2024-08-10)
------------------
* Map and filter now support two argument functions. If such a funtion is provided,
  map/filter is applied on key, value pairs.
* Added exponential function for simple objects.
* Raising a mv to 0.5 is now correctly interpreted as a square root.
  This enables e.g. automatic differentiation.

1.1.2 (2024-11-15)
------------------
* Improved printing, especially for multivector with array or multivector coefficients.
* `pretty_blade` options added to algebra, to allow users to choose the printing of basis blades.
* getattr bugfix

1.2.0 (2024-12-16)
------------------
* Binary operators are now broadcasted across lists and tuples, e.g. `R >> [point1, point2]`.
* Projection (@) and conjugation (>>) are now symbolically optimized by default.
* Matrix reps made with `expr_as_matrix` now have better support for numerical (and multidimensional) multivectors.

1.3.0 (2025-03-10)
------------------
* Added custom basis support! You can now choose your own basis, to reduce the number of sign swaps. E.g. `e31` instead of `e13` for the j quaternion.
* Added `Algebra.fromname` alternative constructor, to initiate popular algebras with optimized bases, identical to `GAmphetamine.js`.
* Codegen has been made 2-15 times faster for basic operators.
* Updated the documentation.

1.3.1 (2025-06-06)
------------------
Bugfix release:
- matrix reps are now correct in all signatures (including custom signatures).
- Fixed setattr discrepancy when trying to set a basis blade with setattr.
- Support copying multivectors

1.4.0 (2025-07-11)
------------------
Massive large algebra improvement!
- In theory up to 36 dimensions are supported*
- Above d > 6 kingdon switches to large algebra mode and attempts to make optimizations
- Exotic algebras like 2DCSGA (R5,3), Mother Algebra (R4,4) and 3DCCGA (R6,3) are no longer out of reach, see teahouse!
- Bugfix: multivectors now take priority over numpy arrays in binary operators even when the numpy array is on the left.
