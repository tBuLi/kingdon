===========
Basic Usage
===========

The most important object in all of :code:`kingdon` is :code:`Algebra`::

    from kingdon import Algebra

An :code:`Algebra` has to be initiated with a number of positive, negative,
and null-dimensions, which are traditionally denoted by :code:`p`, :code:`q` and :code:`r`.
For example, in order to create a 2D Geometric Algebra we can initiate

.. code-block::

    >>> alg = Algebra(p=2, q=0, r=0)
    >>> alg = Algebra(2)  # Equivalent: default value for p, q, r is 0.

The basis blades of the algebra are available in the dictionary :code:`alg.blades`. This can be
added to :code:`globals()` in order to allow for easy access to all the basis blades, and allows
the initiation of multivectors using the basis-blades directly:

.. code-block::

    >>> globals().update(alg.blades)
    >>> x = 2 * e + 1 * e1 - 5 * e2 + 6 * e12

where :code:`e` is the identity element, i.e. :math:`e = 1`.
This way of creating multivectors is particularly useful when writing quick scripts
in an interactive environment.
Let's look at some more general ways of making multivectors, starting with symbolic
multivectors before we go on to numerical multivectors.

Symbolic Multivectors
---------------------

In order to create symbolical multivectors in an algebra, we can call
:class:`~kingdon.algebra.Algebra.multivector` and explicitly pass a :code:`name` argument.
For example, let us create two symbolic vectors :code:`u` and :code:`v` in this algebra:

.. code-block::

    >>> u = alg.multivector(name='u', grades=(1,))
    >>> v = alg.multivector(name='v', grades=(1,))
    >>> u
    u1 𝐞₁ + u2 𝐞₂
    >>> v
    v1 𝐞₁ + v2 𝐞₂

The return type of :meth:`~kingdon.algebra.Algebra.multivector` is an instance of :class:`~kingdon.multivector.MultiVector`.

.. note::
    :code:`kingdon` offers constructors for common types of multivectors, such as the vectors above.
    For example, the vectors above can also be created using :code:`u = alg.vector(name='u')`,
    a scalar by :code:`alg.scalar`, a bivector by :code:`alg.bivector`, a pseudoscalar by
    :code:`alg.pseudoscalar`, and so on.
    More on :ref:`multivector types <Multivector Types>` will follow later.

:class:`~kingdon.multivector.MultiVector`'s support common math operators:

.. code-block::

    >>> u + v
    (u1 + v1) 𝐞₁ + (u2 + v2) 𝐞₂
    >>> u * v
    (u1*v1 + u2*v2) + (u1*v2 - u2*v1) 𝐞₁₂

We also have the inner and exterior "products":

.. code-block::

    >>> u | v
    (u1*v1 + u2*v2)
    >>> u ^ v
    (u1*v2 - u2*v1) 𝐞₁₂

We see that *in the case of vectors* the product is equal to the sum of the inner and exterior,
which is the famous GA relationship :math:`uv = u \cdot v + u \wedge v`.

Since vectors in 2DVGA represent reflections in lines through the origin, we can reflect the
line :code:`v` in the line :code:`u` by using conjugation: :math:`u[v] = - u v u^{-1}`.
This is implemented in kingdon as :code:`u >> v`.

.. code-block::

    >>> u >> v
    (-2*u1*u2*v2 + 2*u2**2*v1 - v1) 𝐞₁ + (2*u1**2*v2 - 2*u1*u2*v1 - v2) 𝐞₂

we see that the result is again a vector, as it should be.

.. warning::
    Kingdon's codegen *assumes* the versor satisfies :math:`u \widetilde{u} = 1`,
    and hence :math:`u^{-1} = \widetilde{u}`,
    for conjugation (:code:`>>`) and projection (:code:`@`).
    This assumption allows kingdon to optimize the generated code even further
    for the common scenario where :math:`u \in \text{Pin}(p,q,r)`.
    However, ensuring that :code:`u` is actually normalized is up to you.
    If you do not want to rely on this assumption, you can also define
    your own :ref:`operators <JIT Expressions>`.

These examples should show that the symbolic multivectors of :code:`kingdon`
make it easy to do symbolic computations. Moreover, we can also use :mod:`sympy` expressions
as values for the multivector:

.. code-block::

    >>> from sympy import Symbol, sin, cos
    >>> t = Symbol('t')
    >>> x = cos(t) * e + sin(t) * e12
    >>> x.normsq()
    (1)

.. note::
    Strings are also automatically converted to symbolics with SymPy.
    So :code:`x` from the example above can also be created as

    .. code-block::

        >>> x = alg.multivector(e='cos(t)', e12='sin(t)')
        >>> x.normsq()
        (1)

More control over basisvectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we do not just want to create a symbolic multivector of a certain grade,
but with specific blades, we can do so by providing the :code:`keys` argument.

.. code-block::

    >>> x = alg.multivector(name='x', keys=('e1', 'e12'))
    >>> x1 𝐞₁ + x12 𝐞₁₂

This can be done either by providing a tuple of strings which indicate which basis-vectors should be present,
or by passing them as integers, i.e. :code:`keys=(0b01, 0b11)` is equivalent to the example above.
Internally, :code:`kingdon` uses the binary representation.

Numerical Multivectors
----------------------
:code:`kingdon` makes no assumptions about the data structures that are passed into a multivector
in order to support ducktyping and customization as much as possible.
This also means it works really well with e.g. :code:`numpy` arrays or :code:`torch` tensors.
(See the :ref:`array section <Array Syntax>` for more specific info.)

For example, to repeat some of the examples above with numerical values, we could do

.. code-block::

    >>> import numpy as np
    >>> uvals, vvals = np.random.random((2, 2))
    >>> u = alg.vector(uvals)
    >>> v = alg.vector(vvals)
    >>> u * v
    (0.1541) + (0.0886) 𝐞₁₂

A big performance bottleneck that we suffer from in Python, is that arrays over objects are very slow. (array of structures.)
So while we could make a numpy array filled with :code:`~kingdon.multivector.MultiVector`'s, this would tank our performance.
:code:`kingdon` gets around this problem by instead accepting numpy arrays as input. (structure of arrays.)
So to make a collection of 3 lines in kingdon, we do

.. code-block::

    >>> import numpy as np
    >>> uvals = np.random.random((2, 3))
    >>> u = alg.vector(uvals)
    >>> u
    ([0.82499172 0.71181276 0.98052928]) 𝐞₁ + ([0.53395072 0.07312351 0.42464341]) 𝐞₂

what is important here is that the first dimension of the array has to have the expected length: 2 for a vector.
All other dimensions are not used by :code:`kingdon`. Now we can reflect this multivector in the :code:`e1` line:

.. code-block::

    >>> v = alg.vector((1, 0))
    >>> v >> u
    ([0.82499172 0.71181276 0.98052928]) 𝐞₁ + ([-0.53395072 -0.07312351 -0.42464341]) 𝐞₂

Despite the different shapes, broadcasting is done correctly in the background thanks to the magic of numpy,
and with vanishing performance penalties as the sizes of the arrays increase.

Multivector Types
-----------------

Every algebra comes with a list of
multivector types, each available as a constructor on the algebra. The k-vectors
(:code:`scalar`, :code:`vector`, :code:`bivector`, ...) are always there, :code:`bireflection` from
:math:`d \geq 2`, and the PGA types :code:`direction`, :code:`evector`, :code:`upoint`, :code:`point`
and :code:`translation` when :math:`r = 1`:

.. code-block::

    >>> pga = Algebra.fromname('3DPGA')
    >>> p = pga.point(name='p'); p
    p032 𝐞₀₃₂ + p013 𝐞₀₁₃ + p021 𝐞₀₂₁ + 1.0 𝐞₁₂₃
    >>> d = pga.direction(name='d'); d
    d032 𝐞₀₃₂ + d013 𝐞₀₁₃ + d021 𝐞₀₂₁
    >>> t = pga.translation(name='t'); t
    1.0 + t01 𝐞₀₁ + t02 𝐞₀₂ + t03 𝐞₀₃

Notice that a point knows its :math:`\mathbf{e}_{123}` coefficient is :code:`1.0`, and a translation
knows its scalar part is :code:`1.0`. These are properties of the type, not values you have to supply:
all three examples above are only three values in memory.
The type-system also enables :code:`kingdon` to generate even shorter code.

Every :math:`k`-vector also has a :math:`d - k` dual, which are available by prepending :code:`pseudo` to the name.
So in 3DPGA a pseudoscalar is a quadvector, and a pseudovector is a trivector:

.. code-block::

    >>> pga.pseudoscalar(name='s')
    s0123 𝐞₀₁₂₃
    >>> pga.pseudovector(name='w')
    w032 𝐞₀₃₂ + w013 𝐞₀₁₃ + w021 𝐞₀₂₁ + w123 𝐞₁₂₃

Results of operations are typed as well, so :code:`kingdon` keeps track of what you are computing with:

.. code-block::

    >>> type(pga.vector(name='u') * pga.vector(name='v'))
    <class 'kingdon.multivector.Bireflection'>
    >>> type(pga.point(name='p') & pga.point(name='q'))
    <class 'kingdon.multivector.Bivector'>

Because points are defined as the dual of the "undual point" :code:`upoint`,
they can be made in a dimension agnostic way:

.. code-block::

    >>> pga.upoint(e1='x').dual()
    x 𝐞₀₃₂ + 1.0 𝐞₁₂₃

You can add your own types with the :code:`extra_types` argument to :class:`~kingdon.algebra.Algebra`,
or replace the standard list entirely with :code:`types`. See :doc:`workings` for how to define one.

Operators
---------

Instances of :mod:`~kingdon.multivector.MultiVector` overload all common Geometric Algebra operators.
Below is an overview:

.. list-table:: Operators
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operation
     - Expression
     - Infix
     - Inline
   * - Geometric product
     - :math:`ab`
     - :code:`a*b`
     - :code:`a.gp(b)`
   * - Inner
     - :math:`a \cdot b`
     - :code:`a|b`
     - :code:`a.ip(b)`
   * - Scalar product
     - :math:`\langle a \cdot b \rangle_0`
     - :code:`(a|b).grade(0)`
     - :code:`a.sp(b)`
   * - Left-contraction
     - :math:`a \rfloor b`
     -
     - :code:`a.lc(b)`
   * - Right-contraction
     - :math:`a \lfloor b`
     -
     - :code:`a.rc(b)`
   * - Outer (Exterior)
     - :math:`a \wedge b`
     - :code:`a ^ b`
     - :code:`a.op(b)`
   * - Regressive
     - :math:`a \vee b`
     - :code:`a & b`
     - :code:`a.rp(b)`
   * - Conjugate :code:`a` by :code:`b` with :math:`\widetilde{b}b = 1`
     - :math:`(-1)^{\text{grade}(b) \text{grade}(a)} b a \widetilde{b}`
     - :code:`b >> a`
     - :code:`b.sw(a)`
   * - Project :code:`a` onto :code:`b` with :math:`\widetilde{b}b = 1`
     - :math:`(a \cdot b) \widetilde{b}`
     - :code:`a @ b`
     - :code:`a.proj(b)`
   * - Commutator of :code:`a` and :code:`b`
     - :math:`a \times b = \tfrac{1}{2} [a, b]`
     -
     - :code:`a.cp(b)`
   * - Anti-commutator of :code:`a` and :code:`b`
     - :math:`\tfrac{1}{2} \{a, b\}`
     -
     - :code:`a.acp(b)`
   * - Sum of :code:`a` and :code:`b`
     - :math:`a + b`
     - :code:`a + b`
     - :code:`a.add(b)`
   * - Difference of :code:`a` and :code:`b`
     - :math:`a - b`
     - :code:`a - b`
     - :code:`a.sub(b)`
   * - Reverse of :code:`a`
     - :math:`\widetilde{a}`
     - :code:`~a`
     - :code:`a.reverse()`
   * - Squared norm of :code:`a`
     - :math:`a \widetilde{a}`
     -
     - :code:`a.normsq()`
   * - Norm of :code:`a`
     - :math:`\sqrt{a \widetilde{a}}`
     -
     - :code:`a.norm()`
   * - Normalize :code:`a`
     - :math:`a / \sqrt{a \widetilde{a}}`
     -
     - :code:`a.normalized()`
   * - Square root of :code:`a`
     - :math:`\sqrt{a}`
     -
     - :code:`a.sqrt()`
   * - Dual of :code:`a`
     - :math:`a*`
     -
     - :code:`a.dual()`
   * - Undual of :code:`a`
     -
     -
     - :code:`a.undual()`
   * - Grade :code:`k` part of :code:`a`
     - :math:`\langle a \rangle_k`
     -
     - :code:`a.grade(k)`


Graphing using :code:`ganja.js`
-------------------------------

:code:`kingdon` supports the :code:`ganja.js` graphing syntax. For those already familiar with
:code:`ganja.js`, the API will feel very similar:

.. code-block::

    >>> alg.graph(0xff0000, u, "u", lineWidth=3)

Running this in a notebook produces:

.. raw:: html

    <div id="kingdon-graph-demo" style="width: 100%; height: 220px; border: 1px solid #ddd; border-radius: 4px;"></div>
    <script>
      // ganja.js registers itself via AMD if a `define.amd` loader (such as the
      // RequireJS instance these docs load for notebook widgets) is present on the
      // page, in which case it never attaches `Algebra` to `window`. So, exactly like
      // kingdon.graph.js does for the notebook widget, we fetch the source and eval
      // it with `define` shadowed to force the plain-global export path instead.
      fetch("https://enki.ws/ganja.js/ganja.js")
        .then((response) => response.text())
        .then((ganja_source) => {
          const ctx = {};
          new Function("const define=1;" + ganja_source).apply(ctx);
          const Algebra = ctx.Algebra;

          Algebra(2, 0, 0, function () {
            var u = 1e1 + 2e2;
            var el = this.graph([0xff0000, u, "u"], {lineWidth: 3});
            el.style.width = "100%";
            el.style.height = "100%";
            document.getElementById("kingdon-graph-demo").appendChild(el);
          });
        });
    </script>

The rules are simple: all positional arguments will be passed on to :code:`ganja.js` as
elements to graph, whereas keyword arguments are passed to :code:`ganja.js` as options.
Hence, the example above will graph the line :code:`u` with :code:`lineWidth = 3`,
and will attach the label "u" to it, and all of this will be red.
Identical to :code:`ganja.js`, valid inputs to :meth:`~kingdon.algebra.Algebra.graph` are (lists of) instances
of :class:`~kingdon.multivector.MultiVector`, strings, and hexadecimal numbers to indicate colors,
or a function without arguments that returns these things.
The strings can be simple labels, or valid SVG syntax.

.. note::
    kingdon supports :code:`ganja.js`'s animation and interactivity in jupyter notebooks,
    `try kingdon in your browser <https://tbuli.github.io/teahouse/>`_ to give it a go!

Large Algebra's
---------------
In theory :code:`kingdon` supports algebra's up to 36D, but your computer might go up in smoke
if you push it that far. In order to make large's algebras feasible, :code:`kingdon` no longer
performs symbolic optimization and caching because this consumes to much memory, and instead
just computes naively.
By default any algebra of :math:`d > 6` is considered large, but it can be forced manually with
the `large` option to :class:`~kingdon.algebra.Algebra` depending on your needs:

.. code-block::

    >>> alg = Algebra(3, large=True)
    >>> alg = Algebra(8, large=False)

For examples of large algebra's, see the OPNS section of the `teahouse <https://tbuli.github.io/teahouse>`_,
which has some demos in the mother algebra :code:`Algebra(4, 4)`, 2D CSGA :code:`Algebra(5, 3)` and
3DCCGA :code:`Algebra(6, 3)`.

Performance Tips
----------------
Because :code:`kingdon` attempts to symbolically optimize expressions the first time they are called, the first
call to any operation takes a bit more time, whereas subsequent calls have extremely good performance.
Note however that even the first execution that includes the symbolic optimization typically
takes on the order of milliseconds, so you probably won't even notice it.

Since `kingdon` v2.2.x the symbolic code generation features a port of `GAMphetamine.js <https://github.com/enkimute/GAmphetamine.js>`_'s
very powerful Common Subexpression Elimination (CSE) algorithm, which results in the most
optimal code known to man. As in, for those cases in which a hand optimized optimum is known,
the CSE optimized code is exactly the same. Moreover, it is *quick* to run.
Praise `Enki <https://github.com/enkimute>`_.

The table below lists multiplications and additions counted in the emitted Python for representative **3DPGA** expressions
when CSE is enabled vs not. Count columns use :code:`muls/adds`.

.. list-table::
   :header-rows: 1
   :widths: 48 14 14

   * - Operation
     - CSE
     - Naive
   * - :code:`R >> p`, :math:`R` even, :math:`p` normalized point
     - :code:`21/18`
     - :code:`72/30`
   * - :code:`R >> d`, :math:`R` even, :math:`d` a direction
     - :code:`18/12`
     - :code:`54/20`
   * - :code:`R >> o`, :math:`R` even, :math:`o` the origin
     - :code:`15/9`
     - :code:`24/12`
   * - :code:`R >> (0.5*e032)`, :math:`R` even, pure :math:`\mathbf{e}_{032}` blade
     - :code:`6/4`
     - :code:`9/6`
   * - :code:`p1 & p2`, join of two normalized points (regressive product)
     - :code:`6/6`
     - :code:`6/10`
   * - :code:`p & l`, normalized point :code:`p`, line :code:`l` (bivector)
     - :code:`9/9`
     - :code:`9/11`
   * - :code:`p1 & p2 & p3`, join of three normalized points
     - :code:`9/12`
     - :code:`30/22`
   * - :code:`(p | P) * P.inv()`, project normalized point :code:`p` on a general (non-normalized) plane :code:`P`
     - :code:`18/12` (+3 divs)
     - :code:`36/20` (+3 divs)
   * - :code:`p @ P`, project normalized point :code:`p` on normalized plane :code:`P`
     - :code:`6/6`
     - :code:`21/15`
   * - :code:`p @ l`, project normalized point :code:`p` on normalized line :code:`l`
     - :code:`12/12`
     - :code:`33/19`
   * - :code:`P >> p`, reflect normalized point :code:`p` in normalized plane :code:`P`
     - :code:`9/7`
     - :code:`33/13`
   * - :code:`l >> p`, reflect normalized point :code:`p` in normalized line :code:`l`
     - :code:`15/12`
     - :code:`48/20`

.. note::
    These counts are reached by `kingdon`'s built-in operators, provided you use the
    :ref:`multivector types <Multivector Types>` that carry the normalization: a :code:`bireflection`
    rather than a generic even multivector, a :code:`point` rather than a generic trivector.
    The :math:`\mathbf{e}_{032}` row is the exception, as it requires
    :func:`~kingdon.algebra.Algebra.compile` with the blade as a constant, since only then
    is its coefficient known at codegen time.

The symbolically optimized code that kingdon produces is already a good starting point for high performance code.
However, there are still several things to be aware of to ensure good performance.

Broadcasting
~~~~~~~~~~~~
Avoid arrays of multivectors, and use multivectors over e.g. :code:`numpy` arrays or :code:`PyTorch`
tensors instead, as shown in :doc:`arrays`.
This ensures the high level overhead of kingdon is paid only once, and we instead delegate
the computation to the underlying datastructures.

JIT Expressions
~~~~~~~~~~~~~~~
To make it easy to optimize larger expressions, :code:`kingdon` offers the :func:`~kingdon.algebra.Algebra.jit`
decorator.

.. code-block::

    >>> alg = Algebra(3, 0, 1)
    >>>
    >>> @alg.jit
    >>> def myfunc(u, v):
    >>>      return u * (u + v)
    >>>
    >>> x = alg.vector(np.random.random(4))
    >>> y = alg.vector(np.random.random(4))
    >>> myfunc(x, y)

Calling the decorated :code:`myfunc` has the benefit that all the numerical computation is done in one single call,
instead of doing each binary operation individually. This has the benefit that all the (expensive) python boilerplate
code is called only once.
Moreover, one can use :code:`@alg.jit(symbolic=True)` to symbolically optimize the expression, similar to how
`kingdon`'s default binary operators work. As we have seen above in the CSE section, this can result in significant
performance improvements. Afterall, the fastest computation is one you do not have to do.

:func:`~kingdon.algebra.Algebra.jit` figures out the symbolic archetypes from the multivectors you call
it with, and caches a compiled function per combination of input types. If you would rather pick the
archetypes yourself, use :func:`~kingdon.algebra.Algebra.compile` directly. It takes the expression
followed by the archetypes and hands you back a :class:`~kingdon.codegen.CompiledExpression`:

.. code-block::

    >>> R = alg.bireflection(name='R', symbolcls=alg.codegen_symbolcls)
    >>> e1 = alg.vector(e1=1)
    >>> rotate_e1 = alg.compile(myfunc, R, e1)

This is worth the extra effort when you know something about the values that :code:`jit` cannot know,
such as the fact that you only ever rotate unit vectors. See
:func:`~kingdon.algebra.Algebra.compile` for a worked example.

Graded
~~~~~~
The first time :code:`kingdon` is asked to perform an operation it hasn't seen before, it performs code generation
for that particular request. Because codegen is a relatively expensive step, it can be beneficial to reduce the number of
times it is needed. An easy way to achieve this is to initiate the :class:`~kingdon.algebra.Algebra` with `graded=True`.
This enforces that :code:`kingdon` does not specialize codegen down to the individual basis blades, but rather only
per grade. This means there are far less combinations that have to be considered and generated.

Numba JIT
~~~~~~~~~
We can enable numba just-in-time compilation by initiating an :class:`~kingdon.algebra.Algebra` with `wrapper=numba.njit`,
which will apply numba's njit decorator to all of kingdon's generated functions.
This comes with a significant cost the first time any operator is called, but subsequent calls to the same operator are
significantly faster.
However, it is worth mentioning that when dealing with :ref:`Numerical Multivectors` over e.g. numpy arrays,
the benefit of using `numba` actually disappears rapidly as the numpy arrays become larger, since then most of the time
is spend in numpy routines anyway.
So you need to experiment carefully if numba is right for you.
