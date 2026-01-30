============
Array Syntax
============
(`interactive example <https://tbuli.github.io/teahouse/lab/index.html?path=2DPGA%2Fex_2dpga_boids.ipynb>`_)

Kingdon was designed to be agnostic to coefficient types, and therefore it is compatible with 
popular array types such as `numpy` or `torch`.
In order to facilitate working with multivectors over arrays, `kingdon` fully 
supports `numpy`'s array indexing and masking syntax, which results in compact yet performant code.

.. note::
These design choices were made in part because working with a multivector of arrays is *much* faster 
than working with an array of multivectors.


The Shape of MultiVectors
-------------------------

The first dimension of a multivector is always the coefficients of the multivector. 
For example, to create a vector in :math:`\mathbb{R}_3` we could do

.. code-block::

    >>> from kingdon import Algebra
    >>> import numpy as np
    >>> 
    >>> alg = Algebra(3)
    >>> xvals = np.random.rand(3)
    >>> x = alg.vector(xvals)
    >>> x.shape
    (3,)

Now if we look at :code:`x.shape`, we see that it is :code:`(3,)`, the same as :code:`xvals.shape`.
However, the length of :code:`x` is :code:`0`:

.. code-block::

    >>> len(x)
    0

This reflects that `x` is a single vector, and therefore not iterable. You might have expected 
iteration over a multivector to iterate over its coefficients, but in `kingdon` multivectors are 
treated as geometric numbers, similar to how complex numbers are treated in complex analysis.

.. note::
    If you need to iterate over the coefficients anyway use `x.map` to map a function on all the 
    coefficients of the multivector.
    For individual access, use attributes access instead, e.g. :code:`x.e1` returns the 
    :math:`\mathbf{e}_1` coefficient.

Now lets make a collection of :math:`N` vectors, and see what changes:

.. code-block::

    >>> N = 5
    >>> xvals = np.random.rand(3, N)
    >>> x = alg.vector(xvals)
    >>> x
    [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864] 𝐞₁ + [0.15599452 0.05808361 0.86617615 0.60111501 0.70807258] 𝐞₂ + [0.02058449 0.96990985 0.83244264 0.21233911 0.18182497] 𝐞₃
    >>> x.shape
    (3, 5)
    >>> len(x)
    5

Hence, we see that the length of the multivector is :code:`5`, and hence we can iterate over the 
multivector to get the individual vectors in `x`:

.. code-block::

    >>> for vector in x:
    >>>     print(vector)
    0.375 𝐞₁ + 0.156 𝐞₂ + 0.0206 𝐞₃
    0.951 𝐞₁ + 0.0581 𝐞₂ + 0.97 𝐞₃
    0.732 𝐞₁ + 0.866 𝐞₂ + 0.832 𝐞₃
    0.599 𝐞₁ + 0.601 𝐞₂ + 0.212 𝐞₃
    0.156 𝐞₁ + 0.708 𝐞₂ + 0.182 𝐞₃

A huge benefit of multidimensional multivectors is that the resulting code is very compact. 
For example, to compute the norm of (all vectors in) `x`, we simply do

.. code-block::

    >>> d = x.norm()
    >>> d
    [0.40624908 1.35939565 1.4067825  0.87453939 0.74750847]

Masking & Indexing
------------------

Now suppose we were only interested in those vectors with :math:`d > 1`, 
we can use numpy's masking syntax to do

.. code-block::

    >>> large_x = x[d.e > 1]
    >>> large_x
    [0.95071431 0.73199394] 𝐞₁ + [0.05808361 0.86617615] 𝐞₂ + [0.96990985 0.83244264] 𝐞₃

First, `d.e` selects the scalar coefficient of the multivector `d`.
Then, `d.e > 1` creates the boolean array `[False  True  True False False]`, indicating 
which elements satisfy the condition. Lastly, `x[d.e > 1]` passes this condition on to the 
multivector coefficients, which are all arrays of shape `(5,)`.
Importantly, `kingdon` passes the thing between the square brackets (`x[...]`) on to the 
coefficients *unseen*, thus enabling not only numpy style indexing and masking, but also 
any other magic your coefficients might do with `__getitem__` overloading.

Mesh
~~~~
As an example of this powerful syntax, consider a mesh defined by two arrays:
- `vertices`: a float array of shape :code:`(N, 4)` that defines the :math:`(x, y, z, 1)` homogenous 
  coordinates of :math:`N` vertices.
- `faces`: an integer array of shape :code:`(M, 3)` that defines the topology of the faces of the mesh.
  The integers are indices into the `vertices` array, and hence in the range :math:`[0, N)`.

In order to convert the vertices to multivectors (in 3DPGA), we need to transpose `vertices` such that the 
`x,y,z`-coordinates can be matched up with the :math:`\mathbf{e}_i^*` directions:

.. code-block::

    >>> pga3d = Algebra(3, 0, 1)
    >>> v = pga3d.vector(vertices.T).dual()
    >>> v.shape
    (4, N)

Suppose we now want to alternativelly have a datastructure of shape :math:`(4, M, 3)`, 
which explicitelly contains the coordinates of the vertices for every face, similar to how 
`STL files <https://en.wikipedia.org/wiki/STL_(file_format)#ASCII>`_ are structured.
This is now as simple as 

.. code-block::

    >>> facets = v[faces]
    >>> facets.shape
    (4, M, 3)

In order to compute the area and (signed) volume of the mesh we can use numpy's indexing 
syntax to first create all planes,

.. code-block::

    >>> planes = facets[..., 0] & facets[..., 1] & facets[..., 2]
    >>> planes.shape
    (4, M)

from which we can then compute the area:

.. code-block::

    >>> areas = planes.norm()
    >>> area = areas.map(np.sum)

and the volume:

.. code-block::

    >>> volume = np.sum(planes.e0)

Yes, we just computed the signed volume directly from the coefficient of :math:`\mathbf{e}_0`! 
This coefficient is the signed volume a plane makes with the area, and by adding this up for all 
planes we find the volume of the mesh. For more info, see `this paper <https://arxiv.org/abs/2511.08058>`_.