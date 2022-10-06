"""
This module contains support functions to turn
:class:`~kingdon.algebra.MultiVector`'s into matrices and vise versa.

This follows the approach outlined in
Graded Symmetry Groups: Plane and Simple, section 10.
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
    return np.array([[Ri[j, 0] for Ri in Rs] for j in range(len(Rs[0]))])


def matrix_rep(p=0, n=0, z=0):
    d = p+n+z
    I = I2
    P = P2
    Z = Z2
    N = N2
    Ip = Ip2

    # Store all the signature matrices needed for the E_i
    Ss = [Z for _ in range(z)]
    Ss.extend([P for _ in range(p)])
    Ss.extend([N for _ in range(n)])
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
