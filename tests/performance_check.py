from kingdon import Algebra
import timeit
import numpy as np
from math import comb
from itertools import product, chain
import clifford as cf
from collections import defaultdict
import cProfile, pstats
from numba import njit, vectorize

if __name__ == "__main__":
    num_iter = 1000000
    num_rows = 1
    d = 3
    cliff = True
    king = True
    king_profile = False

    # shape_b = (comb(d, 2), num_rows) if num_rows != 1 else comb(d, 2)
    # shape_v = (comb(d, 1), num_rows) if num_rows != 1 else comb(d, 1)
    shape_b = (2 ** d, num_rows) if num_rows != 1 else 2 ** d
    shape_v = (2 ** d, num_rows) if num_rows != 1 else 2 ** d

    if king:
        print("Kingdon", end='\n\n')
        operations = [
            'b*v',
            'b.cp(v)',
            'b >> v',
            'b @ v',
            'b^v', 'b|v', 'b+v', 'b-v',
            'b.inv()',
            'alg.inv(b)',
            'b / v',
            '~b'
        ]
        times = defaultdict(list)
        for operation in operations:
            print(operation)
            for cse, numba in product([False, True], repeat=2):
                alg = Algebra(d, numba=numba, cse=cse)
                # print(alg)
                bvals = np.random.random(shape_b)
                vvals = np.random.random(shape_v)
                # b = alg.bivector(bvals)
                # v = alg.vector(vvals)
                b = alg.multivector(bvals)
                v = alg.multivector(vvals)
                # prepare, does cse and jit.
                init = timeit.timeit(operation, number=1, globals=globals())
                # init = float('inf')
                t = timeit.timeit(operation, number=num_iter, globals=globals())
                print(f'setup with {cse=} & {numba=} took {init:.2E}. Performed {num_iter} iterations, per iteration: {t/num_iter:.2E} sec')
                times[operation].append([cse, numba, init, t/num_iter])

        print('Kingdon Best times:', end='\n\n')
        for operation, timings in times.items():
            cse, numba, init, t = min(timings, key=lambda x: x[-1])
            print(f'{operation}, Setup took {init:.2E}, per iteration: {t:.2E}, {cse=}, {numba=}')

    if cliff:
        print()
        print("Clifford", end='\n\n')
        operations = ['b*v', 'b.commutator(v)', 'b*v*(~b)', '(b|v)*(~v)', 'b ^ v', 'b | v', 'b + v', 'b - v', 'b.inv()', 'b / v', '~b']
        for operation in operations:
            layout, blades = cf.Cl(d)
            # v = (layout.randomMV())(1)
            # b = (layout.randomMV())(2)
            v = (layout.randomMV())
            b = (layout.randomMV())
            init = timeit.timeit(operation, number=1, globals=globals())
            t = timeit.timeit(operation, number=num_iter, globals=globals())
            print(f"{operation}. Setup took {init:.2E}. Performed {num_iter} iterations, per iteration: {t/num_iter:.2E}.")

    if king_profile:
        alg = Algebra(d, numba=False, cse=True)
        bvals = np.random.random(shape_b)
        vvals = np.random.random(shape_v)
        b = alg.multivector(bvals)
        v = alg.multivector(vvals)
        # print(b.cp(v))
        operation = 'b.sw(v)'
        init = timeit.timeit(operation, number=1, globals=globals())
        print('init', init)
        # alg.nonop(b)
        filename = 'restats6'
        prof = cProfile.run(f'for _ in range({num_iter}): {operation}', filename)

        ps = pstats.Stats(filename).sort_stats('tottime')
        ps.print_stats()

