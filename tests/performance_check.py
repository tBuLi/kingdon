from kingdon import Algebra
import timeit
import numpy as np
from math import comb
from itertools import product, chain
import clifford as cf
import cProfile, pstats
from numba import njit, vectorize

num_iter = 10000
num_rows = 1
d = 3
# shape_b = (comb(d, 2), num_rows) if num_rows != 1 else comb(d, 2)
# shape_v = (comb(d, 1), num_rows) if num_rows != 1 else comb(d, 1)
shape_b = (2**d, num_rows) if num_rows != 1 else 2**d
shape_v = (2**d, num_rows) if num_rows != 1 else 2**d
#
print("Kingdon", end='\n\n')
operations = ['b*v', 'b.cp(v)', 'b.conj(v)', 'b^v', 'b|v']
for operation in operations:
    print(operation)
    for cse, numba in product([False, True], repeat=2):
        alg = Algebra(d, numba=numba, cse=cse)
        # print(alg)
        bvals = np.random.random(shape_b)
        vvals = np.random.random(shape_v)
        # b = alg.bivector(name='B', vals=bvals)
        # v = alg.vector(name='v', vals=vvals)
        b = alg.multivector(name='B', vals=bvals)
        v = alg.multivector(name='v', vals=vvals)
        # prepare, does cse and jit.
        init = timeit.timeit(operation, number=1, globals=globals())
        # init = float('inf')
        t = timeit.timeit(operation, number=num_iter, globals=globals())
        print(f'setup with cse={cse} & numba={numba} took {init:.1E}. Performed {num_iter} iterations, per iteration: {t/num_iter:.1E} sec')

print()
print("Clifford", end='\n\n')
layout, blades = cf.Cl(d)
# v = (layout.randomMV())(1)
# b = (layout.randomMV())(2)
v = (layout.randomMV())
b = (layout.randomMV())
operations = ['b*v', 'b.commutator(v)', 'b*v*(~b)', 'b ^ v', 'b | v']
for operation in operations:
    init = timeit.timeit(operation, number=1, globals=globals())
    t = timeit.timeit(operation, number=num_iter, globals=globals())
    print(f"clifford {operation}. Setup took {init:.1E}. Performed {num_iter} iterations, per iteration: {t/num_iter:.1E} sec.")

# alg = Algebra(3, 0, 1)
# bvals = np.random.random(shape_b)
# vvals = np.random.random(shape_v)
# b = alg.bivector(name='B', vals=bvals)
# v = alg.vector(name='v', vals=vvals)
# # print(b.cp(v))
# init = timeit.timeit('b*v', number=1, globals=globals())
# print('init', init)
# prof = cProfile.run(f'for _ in range({num_iter}): b*v', 'restats')
#
# ps = pstats.Stats('restats').sort_stats('tottime')
# ps.print_stats()

# alg = Algebra(3, 0, 1)
# bvals = np.random.random(shape_b)
# vvals = np.random.random(shape_v)
# b = alg.multivector(name='B', vals=bvals)
# v = alg.multivector(name='v', vals=vvals)
# print(id(bvals[0]), id(b[0]))
# w = b*v
# print(w)
