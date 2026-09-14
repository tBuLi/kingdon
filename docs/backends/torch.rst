=======
PyTorch
=======

Opt in to the :code:`torch` backend, and a multivector over torch coefficients can be handed to a
:code:`torch.nn` module directly:

.. code-block::

    >>> import torch
    >>> from kingdon import Algebra
    >>>
    >>> alg = Algebra.fromname("3DPGA", backend="torch")
    >>> points = alg.point(torch.randn(3, 32, 4))
    >>> points.shape
    Point[(32, 4)]
    >>> torch.nn.Sequential(torch.nn.Linear(4, 10), torch.nn.GELU())(points).shape
    Point[(32, 10)]

:code:`backend="torch"` is what imports :mod:`kingdon.torch_backend`, and importing that is what
puts :code:`__torch_function__` on a multivector so that :code:`torch` can interact with it.
The backend also sets a :func:`~kingdon.torch_backend.values_asarray` that keeps the
coefficients of a multivector in a single tensor.
You can overwrite that by passing your own :code:`values_asarray`, but the one that ships with :code:`kingdon`
was designed to be compatible with :func:`torch.compile` so it is probably the right choice.


Why it works
============

A multivector over an array of shape :code:`(blades, ..., channels)` has shape
:code:`(..., channels)`: the first axis of :code:`mv.values()` is the blade axis, and
:attr:`~kingdon.multivector.MultiVector.shape` does not expose it.

A torch function is handed :code:`mv.values()`, and its result becomes the coefficients of the
multivector that comes back -- unless the multivector has an operation of that name itself, see
`The operators`_. That is all of it, and it is enough because the blade axis *leads*:
everything :code:`torch.nn` is built out of treats leading axes as batch axes. A :code:`Linear`
with weights of :code:`(channels, channels_out)` therefore acts on all blades in a single call, and
so do the activation functions, :code:`LayerNorm` and :code:`Dropout`. Nothing is translated, and
kingdon keeps no list of which functions are safe.

.. note::
    Some torch operations might attempt to manipulate the blade dimension, which is not allowed because this
    would interfere with the geometry.

.. code-block::

    >>> torch.sum(points, -1).shape       # sum the channels, allowed.
    Point[(32,)]
    >>> torch.sum(points, 0)              # sum the blades, not allowed.
    TypeError: sum left (32,) where the 3 blades of this Point were, so it addressed the blade axis.

Beyond that you may do to the coefficients whatever you like, sensible or not.

.. note::
    Torch integration makes it very easy to write GA based equivariant neural networks, but it is
    also a loaded footgun: do not assume torch's builtin modules are equivariant! Typically they are not!
    For equivariant modules, see https://tbuli.github.io/rotorch/.

Addressing the axes with einops
===============================

To address the axes of the multivector rather than of its coefficients, use :code:`einops`, whose
patterns refer to :code:`mv.shape` and leave the blade axis out of it. See :doc:`../arrays`.

.. code-block::

    >>> from einops import reduce, rearrange, pack, einsum
    >>>
    >>> reduce(points, 'points channels -> channels', 'sum').shape
    Point[(4,)]
    >>> rearrange(points, 'points channels -> channels points').shape
    Point[(4, 32)]
    >>> pack([points, points], '* channels')[0].shape
    Point[(64, 4)]
    >>> einsum(points, torch.randn(4, 7), 'points channels, channels out -> points out').shape
    Point[(32, 7)]

The operators
=============

There is one rule for when a torch name does not mean the coefficients:

.. note::
    If a multivector has an operation of that name, torch's name means the multivector's.

However torch spells an operator, it is therefore handed to the algebra, and on whichever side of it
the multivector sits: :code:`tensor | mv` is the inner product as much as :code:`mv | tensor` is.
This allows the Cayley table of the geometric algebra to be "injected" into torch code without any
changes having to be made on the torch side.

================  =========================  ================================  ====================
operator          torch spells it            kingdon                           which is
================  =========================  ================================  ====================
:code:`x + y`     :func:`torch.add`          :func:`~kingdon.operators.add`     addition
:code:`x - y`     :func:`torch.sub`          :func:`~kingdon.operators.sub`     subtraction
:code:`-x`        :func:`torch.neg`          :func:`~kingdon.operators.neg`     negation
:code:`x * y`     :func:`torch.mul`          :func:`~kingdon.operators.gp`      geometric product
:code:`x / y`     :func:`torch.div`          :func:`~kingdon.operators.div`     division
:code:`x @ y`     :func:`torch.matmul`       :func:`~kingdon.operators.proj`    projection
:code:`x | y`     :code:`Tensor.__or__`      :func:`~kingdon.operators.ip`      inner product
:code:`x ^ y`     :code:`Tensor.__xor__`     :func:`~kingdon.operators.op`      outer product
:code:`x & y`     :code:`Tensor.__and__`     :func:`~kingdon.operators.rp`      regressive product
:code:`x >> y`    :code:`Tensor.__rshift__`  :func:`~kingdon.operators.sw`      sandwich product
================  =========================  ================================  ====================

The aliases go along: :func:`torch.subtract`, :func:`torch.multiply`, :func:`torch.divide`,
:func:`torch.true_divide` and :func:`torch.negative`. The last four operators have no torch function
of their own, only the dunder, so those are reached by :code:`tensor ^ mv` alone; with the
multivector on the left python never asks torch in the first place.

The same rule reaches the operations that torch has a name for but no operator. A multivector has
an :code:`exp`, so :func:`torch.exp` is the exponential *of the multivector*:

==================  =============================================  ==================================
torch               kingdon                                        which is
==================  =============================================  ==================================
:func:`torch.exp`   :meth:`~kingdon.multivector.MultiVector.exp`   the exponential of a simple element
:func:`torch.sqrt`  :meth:`~kingdon.multivector.MultiVector.sqrt`  the root of a Study number
:func:`torch.norm`  :meth:`~kingdon.multivector.MultiVector.norm`  the norm under the metric
==================  =============================================  ==================================

.. code-block::

    >>> B = alg.bivector(e12=torch.tensor(0.3))
    >>> torch.exp(B)                  # the exponential of the multivector: a rotor
    tensor(0.9553) + tensor(0.2955) 𝐞₁₂
    >>> torch.exp(B.values())         # the exponential of its coefficients
    tensor([1.3499])

A name a multivector does not have is handed the coefficients as ever, so :code:`torch.relu(mv)` is
the relu of every one of them. And :code:`mv.values()` is always there when the coefficients are
what you mean.

Gradients
=========

Gradients flow through the coefficients, so a multivector may be built and taken apart inside a
:code:`forward`. Register the raw tensor as the :class:`~torch.nn.Parameter`; a multivector held as
a module attribute is not visited by :code:`torch.nn.Module`, so its coefficients would not be
registered.

.. code-block::

    >>> class GALayer(torch.nn.Module):
    ...     def __init__(self, n_in, n_out):
    ...         super().__init__()
    ...         self.lin = torch.nn.Linear(n_in, n_out)
    ...         self.bivector = torch.nn.Parameter(torch.randn(3) * 0.1)
    ...
    ...     def forward(self, p):
    ...         b = self.bivector
    ...         R = alg.bivector(e12=b[0], e13=b[1], e23=b[2]).exp()   # a learned rotor
    ...         return torch.relu(self.lin(R >> p))                   # rotate, mix, activate
    >>>
    >>> out = GALayer(4, 10)(points)
    >>> out.shape
    Point[(32, 10)]
    >>> out.values().sum().backward()

:code:`mv.values()` is one tensor, ready to reduce into a loss. Tensor methods are available on the
multivector too, since :code:`mv.detach()` and :code:`mv.to(device)` go the same way as the free
functions:

.. code-block::

    >>> points.detach().to(torch.float64).dtype
    torch.float64

Finally, :code:`wrapper=torch.compile` applies :code:`torch.compile` to every function kingdon
generates, after its own symbolic optimization and cse:

.. code-block::

    >>> alg = Algebra.fromname("3DPGA", backend="torch", wrapper=torch.compile)

A module that takes and gives back multivectors can also be exported whole, since the backend
registers the multivector types with :code:`torch.utils._pytree`:

.. code-block::

    >>> model = torch.nn.Sequential(torch.nn.Linear(4, 10), torch.nn.GELU())
    >>> torch.export.export(model, (points,)).module()(points).shape
    Point[(32, 10)]

.. note::
    An operation that has to look at the *value* of a coefficient cannot be exported, since export
    guards on shapes rather than values. :meth:`~kingdon.multivector.MultiVector.exp` is one of
    those: it checks that the element squares to a scalar, which for array coefficients is a
    question about their values, so a layer that builds a rotor inside its :code:`forward` exports
    no further than :code:`GuardOnDataDependentSymNode`.
