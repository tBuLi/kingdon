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
