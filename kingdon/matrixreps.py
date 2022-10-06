"""
This module contains support functions to turn
:class:`~kingdon.algebra.MultiVector`'s into matrices.

This follows the approach outlined in
Graded Symmetry Groups: Plane and Simple, section 10.
See the paper for more details.
"""
from functools import reduce
from itertools import combinations
import numpy as np


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
    return np.array([[Ri[j, 0] for Ri in Rs] for j in range(len(Rs[0]))])


def matrix_rep(p=0, q=0, r=0):
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
    d = p+q+r
    I = I2
    P = P2
    Z = Z2
    N = N2
    Ip = Ip2

    # Store all the signature matrices needed for the E_i
    Ss = [Z for _ in range(r)]
    Ss.extend([P for _ in range(p)])
    Ss.extend([N for _ in range(q)])
    # Construct the matrix reps for the E_i from E_d to E_0
    Es = []
    for i, Si in enumerate(Ss):
        mats = [I for _ in range(i)]
        mats.append(Si)
        mats.extend([Ip for _ in range(d - i - 1)])
        Es.append(reduce(np.kron, mats, 1))
    Es = list(reversed(Es))

    Rs = Es.copy()
    Iden = reduce(np.kron, [I for _ in range(d)])
    Rs.insert(0, Iden)

    # Extend Rs with the higher order basis-blades.
    for i in range(2, d+1):
        Rs_grade_i = [reduce(lambda x, y: x @ y, comb)
                      for comb in combinations(Es, r=i)]
        Rs.extend(Rs_grade_i)

    O = ordering_matrix(Rs)
    return [O@Ri@O.T for Ri in Rs]
