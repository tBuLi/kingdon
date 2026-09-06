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
* matrix reps are now correct in all signatures (including custom signatures).
* Fixed setattr discrepancy when trying to set a basis blade with setattr.
* Support copying multivectors

1.4.0 (2025-07-11)
------------------
Massive large algebra improvement!
* In theory up to 36 dimensions are supported*
* Above d > 6 kingdon switches to large algebra mode and attempts to make optimizations
* Exotic algebras like 2DCSGA (R5,3), Mother Algebra (R4,4) and 3DCCGA (R6,3) are no longer out of reach, see teahouse!
* Bugfix: multivectors now take priority over numpy arrays in binary operators even when the numpy array is on the left.

2.0.0 (2026-03-03)
------------------
* Length of a multivector is now defined such that multivectors are sequences if their coefficients are arrays.
  This allows users to iterate over e.g. point clouds naturally.
* Improved documentation for array syntax.
* Large algebra performance improvements.

2.1.0 (2026-03-12)
------------------
* The conjugation operator ``>>`` now implements the twisted Clifford-Lipschitz action, meaning that it correctly implements sign flips on the basis of grade. Moreover, it now assumes that :math:`R\tilde{R}=1`, which allows us to carry out performance improvements in the future. See #123. This also means that if this not what you desire, you should implement your own sandwich operator.
* The projection ``@`` now assumes that the second argument is a versor (k-reflection).
* Graphs can now be updated on the fly with the ``GraphWidget.update`` method.
* The options provided to ``GraphWidget`` now also include a ``style`` argument, which allows you to set arbitrary arguments on ``canvas.style`` and so the graphs can now be customized to a much larger degree.

2.1.1 (2026-04-14)
------------------
Bugfix: width and height should be direct options to Algebra.graph

3.0.0 (2026-09-04)
------------------
* Kingdon now has a type system, inspired by ``GAmphetamine.js``. Multivector types such as ``point``, ``direction``, ``translation`` and ``bireflection`` are defined by a ``layout``: either a dict of basis blades, or a classmethod holding a GA expression, from which kingdon derives which coefficients are free and which are structural constants. Constants are no longer stored or computed with, so the generated code is shorter: the counts in the CSE table of the docs are now reached by the built-in operators. Register your own types with the ``extra_types`` (or ``types``) argument to ``Algebra``.
* The ``Algebra`` option ``graded`` has been renamed to ``full_layout``, since it now enforces that every multivector carries the full layout of its type.
* ``Algebra.register`` has been deprecated in favor of ``Algebra.add_operator``, which is what it really does: it adds a new operator to the algebra. The old name still works but raises a ``FutureWarning``.
* ``MultiVector`` is no longer callable. To evaluate an expression for numerical values, build the multivector numerically or compile the expression with ``Algebra.add_operator``/``Algebra.compile``.
* Einops is now supported on Multivectors, allowing users to write einops expressions on multivector coefficients and seamlessly mix them with kingdon operations. See #46.
* GAmphetamine.js inspired Common Subexpression Elimination (CSE) has been implemented, resulting in code that is as fast as hand optimized code for known test cases.
* The ``Algebra.add_operator`` decorator now uses the new CSE by default, so compiled functions are now even faster without any changes to user code.
* ``Algebra.compile`` is no longer a decorator: it now takes the expression followed by the symbolic multivectors to compile it for, and returns a ``CompiledExpression``.
* ``Algebra.graph`` now accepts multivectors whose coefficients are arrays, so meshes and point clouds are a single subject instead of thousands. The shape of the multivector decides what is drawn: ``(N,)`` gives ``N`` separate elements, ``(N, 2)`` line segments and ``(N, 3)`` filled triangles. The coefficients are sent to ``ganja.js`` as a single binary buffer, which is much faster than graphing the elements one by one. See the "Meshes and point clouds" section of the docs.
* ``RationalPolynomial`` now has a ``diff`` method similar to that of SymPy.
* The ``set`` method of Multivectors can be used within compiled functions to update coefficients in-place instead of returning a multivector.
* Large algebras (``large=True``) have no multivector types: every multivector is a plain ``MultiVector``. The k-vector constructors such as ``alg.vector`` and ``alg.pseudovector`` are still but they simply construct a ``MultiVector`` of that grade. ``Algebra.add_operator`` requires ``symbolic=True`` in a large algebra.
