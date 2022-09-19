from kingdon import Algebra
import timeit
import numpy as np
from math import comb
from itertools import product, chain
import clifford as cf
import cProfile

num_iter = 10000
num_rows = 1
d = 6
shape_b = (comb(d, 2), num_rows) if num_rows != 1 else comb(d, 2)
shape_v = (comb(d, 1), num_rows) if num_rows != 1 else comb(d, 1)

# print("Kingdon", end='\n\n')
# operations = ['b*v', 'b.cp(v)', 'b >> v']
# for operation in operations:
#     print(operation)
#     for cse, numba in product([False, True], repeat=2):
#         alg = Algebra(d, numba=numba, cse=cse)
#         # print(alg)
#         bvals = np.random.random(shape_b)
#         vvals = np.random.random(shape_v)
#         b = alg.bivector(name='B', vals=bvals)
#         v = alg.vector(name='v', vals=vvals)
#         # b = alg.bivector(name='B', vals={3: bvals[0]})
#         # v = alg.vector(name='v', vals={1: vvals[0]})
#         # prepare, does cse and jit.
#         init = timeit.timeit(operation, number=1, globals=globals())
#         t = timeit.timeit(operation, number=num_iter, globals=globals())
#         print(f'setup with cse={cse} & numba={numba} took {init:.1E}. Performed {num_iter} iterations, per iteration: {t/num_iter:.1E} sec')
#
# print()
# print("Clifford", end='\n\n')
# layout, blades = cf.Cl(d)
# v = layout.randomMV()(1)
# b = layout.randomMV()(2)
# operations = ['b*v', 'b.commutator(v)', 'b*v*(~b)']
# for operation in operations:
#     t = timeit.timeit(operation, number=num_iter, globals=globals())
#     print(f"clifford {operation}. Performed {num_iter} iterations, per iteration: {t/num_iter:.1E} sec.")

alg = Algebra(3, 0, 1, cse=True)
bvals = np.random.random(shape_b)
vvals = np.random.random(shape_v)
b = alg.bivector(name='B', vals=bvals)
v = alg.vector(name='v', vals=vvals)

# prof = cProfile.run('b*v')
b*v
# print(prof)
# prof = cProfile.run('for _ in range(100000): b*v')
# print(prof)

def _lambdifygenerated(a3, a5, a9, a6, a10, a12, b1, b2, b4, b8):
    return (-a3*b1 + a6*b4, a3*b2 + a5*b4, a3*b4 - a5*b2 + a6*b1, a10*b1 + a3*b8 - a9*b2, -a5*b1 - a6*b2, a12*b1 + a5*b8 - a9*b4, -a10*b2 - a12*b4 - a9*b1, -a10*b4 + a12*b2 + a6*b8)
#
# args = chain(b.vals.values(), v.vals.values())
# prof = cProfile.run('for _ in range(100000): _lambdifygenerated(*chain(b.vals.values(), v.vals.values()))')
# print(prof)

b = alg.bivector(vals={3: 2})
v = alg.vector(vals={1: 3})
print(b.cp(v))
