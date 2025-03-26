"""
This module contains support functions to turn
:class:`~kingdon.multivector.MultiVector`'s into matrices.

This follows the approach outlined in
Graded Symmetry Groups: Plane and Simple, section 10.
See the paper for more details.
"""
import itertools
import string
from functools import reduce
from itertools import combinations
from typing import Callable
import numpy as np
import sympy


I2 = np.array([[1,0], [0,1]])
Ip2 = np.array([[1,0], [0,-1]])
P2 = np.array([[0,1], [1,0]])
N2 = np.array([[0,1], [-1,0]])
Z2 = np.array([[0,0], [1,0]])


def ordering_matrix(Rs):
    """
    Matrix reps are determined up to similarity transform.
    But not all similarity transforms are equal.
    This function creates the one similarity transform that gives all
    entries in the first column of the matrix a positive sign.
    In doing so, matrix-matrix multiplication is identical to matrix-vector
    multiplication.

    :param Rs: sequence of matrix reps for all the basis blades of an algebra.
    :return: The similarity transform to beat all similarity transforms.
    """
    columns = [Ri[:, 0] for Ri in Rs]
    return np.vstack(columns)


def matrix_rep(p=0, q=0, r=0, signature=None):
    """
    Create the matrix reps of all the basis blades of an algebra.
    These are selected such that the entries in the first column
    of the matrix have positive sign, and thus matrix-matrix multiplication
    is identical to matrix-vector multiplication.

    :param p: number of positive dimensions.
    :param q: number of negative dimensions.
    :param r: number of null dimensions.
    :return: sequence of matrix reps for the basis-blades.
    """
    d = p + q + r
    I = I2
    P = P2
    Z = Z2
    N = N2
    Ip = Ip2

    # Store all the signature matrices needed for the E_i
    SsR = [Z for _ in range(r)]
    SsP = [P for _ in range(p)]
    SsN = [N for _ in range(q)]
    if signature is not None:
        Ss = []
        for s in signature:
            if s == 0:
                Ss.append(SsR.pop(0))
            elif s == 1:
                Ss.append(SsP.pop(0))
            elif s == -1:
                Ss.append(SsN.pop(0))
    else:
        Ss = [*SsR, *SsP, *SsN]
    # Construct the matrix reps for the E_i from E_d to E_0
    Es = []
    for i, Si in enumerate(Ss):
        mats = [I for _ in range(i)]
        mats.append(Si)
        mats.extend([Ip for _ in range(d - i - 1)])
        Es.append(reduce(np.kron, mats, 1))
    Es = list(Es)

    Rs = Es.copy()
    Iden = reduce(np.kron, [I for _ in range(d)])
    Rs.insert(0, Iden)

    # Extend Rs with the higher order basis-blades.
    for i in range(2, d+1):
        Rs_grade_i = [reduce(lambda x, y: x @ y, comb)
                      for comb in combinations(Es, r=i)]
        Rs.extend(Rs_grade_i)

    O = ordering_matrix(Rs)
    return [O @ Ri @ O.T for Ri in Rs]


def expr_as_matrix(expr: Callable, *inputs, res_like: "MultiVector" = None):
    """
    This represents any GA expression as a matrix. To illustrate by example, we might want to
    represent the multivector equation y = R >> x as a matrix equation y = Ax.
    (If the multivector `x` is of pure grade, the matrix `A` will be in irrep of the transformation.)
    To obtain A, call this function as follows::

        alg = Algebra(3, 0, 1)
        R = alg.evenmv(name='R')
        x = alg.vector(name='x')
        A, y = expr_as_matrix(lambda R, x: R >> x, R, x)

    In order to build the matrix rep the input `expr` is evaluated, so make sure the inputs
    to the expression are given in the correct order.
    The last of the positional arguments is assumed to be the vector x in the linear equation y = Ax,
    and is *assumed to be symbolic*.
    The other arguments can also be numeric or e.g. a multidimensional array/torsor, in which case the
    returned matrix A will be numerical as well. This can e.g. be used to easily generate the matrix
    representations of a given (dual-)quaternion::

        >>> alg = Algebra(3, 0, 1)
        >>> R = alg.evenmv(e12=0.25*numpy.pi).exp()
        >>> x = alg.vector(name='x')
        >>> A, y = expr_as_matrix(lambda R, x: R >> x, R, x)
        >>> A
        [[ 1.  0.  0.  0.]
         [ 0.  0.  1.  0.]
         [ 0. -1.  0.  0.]
         [ 0.  0.  0.  1.]]

    :expr: Callable representing a valid GA expression.
        Can also be a :class:`~kingdon.operator_dict.OperatorDict`.
    :inputs: All positional arguments are consider symbolic input arguments to `expr`. The last of these is assumed to
        represent the vector `x` in `y = Ax`.
    :res_like: (optional) multivector corresponding to the desired output. If None, then the full output is returned.
        However, if only a subsegment of the output is desired, provide a multivector with the desired shape.
        In the example above setting, `res_like = alg.vector(e1=1)` would mean only the e1 component of the matrix
        is returned. This does not have to be a symbolic multivector, only the keys are checked.
    :return: This function returns the matrix representation, and the result of applying the expression to the input.
        If at least one of the inputs other than the last one is symbolic, the result will be a sympy symbolic matrix.
        Otherwise, the result will be a numpy array.
    """
    *rest, x = inputs
    alg = x.algebra
    numerical = all(not r.issymbolic for r in rest)
    if numerical and any(len(r.shape) > 1 for r in rest):  # Only do this for multidimensional arrays
        symbolic_rest = [alg.multivector(name=string.ascii_uppercase[i], keys=mv.keys()) for i, mv in enumerate(rest)]
        symbolic_inputs = [*symbolic_rest, x]
        A, y = expr_as_matrix(expr, *symbolic_inputs, res_like=res_like,)
        symbols2values = dict(itertools.chain(*(zip(smv.values(), mv.values()) for smv, mv in zip(symbolic_rest, rest))))
        func = sympy.lambdify(symbols2values.keys(), A, modules={'ImmutableDenseMatrix': list})
        kwargs = {str(k): v for k, v in symbols2values.items()}
        A = func(**kwargs)  # TODO: vectorize this call correctly
        symbols2values.update({v: v for v in x.values()})
        y = y(**{str(k): v for k, v in symbols2values.items() if k in y.free_symbols})
        return A, y

    y = expr(*inputs)
    if res_like is not None:
        y = alg.multivector({k: sympy.sympify(getattr(y, alg.bin2canon[k])) for k in res_like.keys()})

    A = sympy.zeros(len(y), len(x)) if not numerical else np.zeros((len(y), len(x)))
    for i, (blade_y, yi) in enumerate(y.items()):
        cv = sympy.collect(yi.expand(), x.values())
        for j, (blade_x, xj) in enumerate(x.items()):
            A[i, j] = cv.coeff(xj)
    return A, y
