import numpy as np
from egglog import *
import time

from rewrites.EGraph import egraph
from rewrites import Matrix
from compiler.compiler import compile
from compiler.compiler import comp_timer
from utils.sparse_randn import sparse_randn

egraph.push()

"""
FAVORABLE EXAMPLE WHERE THE OPTIMIZATION RESULTS IN A FASTER COMPUTATION
"""
print("FAVORABLE EXAMPLE")

data = {
    'a': sparse_randn((64000, 900), 0.99),
    'b': sparse_randn((900, 2560), 0.99),
}

matrices = {
    'a': Matrix('a', 64000, 800, 0.99),
    'b': Matrix('b', 800, 2560, 0.99),
}

expr = egraph.let("expr", matrices['a']@matrices['b'])
result = compile(expr, egraph, data, DEBUG=True)
comp_timer.report()

print("\nPerforming naive operation...")
start = time.perf_counter()
result_naive = data['a'] @ data['b']
end = time.perf_counter()
print(f'Naive version ran in: {end-start:.6f}')

if np.allclose(result.toarray(), result_naive, rtol=1e-5, atol=1e-8):
    print("\nResults are matching !")
else:
    print("Results do NOT match );")

egraph.pop()

egraph.push()

"""
UNFAVORABLE EXAMPLE WHERE THE OPTIMIZATION RESULTS IN A MUCH LONGER COMPUTATION TIME
DUE TO THE STORAGE FORMAT CONVERSION OF THE MATRICES
"""

print("\n\nUNFAVORABLE EXAMPLE")

data = {
    'a': sparse_randn((64000, 900), 0.79),
    'b': sparse_randn((900, 2560), 0.83),
}

matrices = {
    'a': Matrix('a', 64000, 800, 0.79),
    'b': Matrix('b', 800, 2560, 0.83),
}

expr = egraph.let("expr", matrices['a']@matrices['b'])
result = compile(expr, egraph, data, DEBUG=True)
comp_timer.report()

print("\nPerforming naive operation...")
start = time.perf_counter()
result_naive = data['a'] @ data['b']
end = time.perf_counter()
print(f'Naive version ran in: {end-start:.6f}')

if np.allclose(result.toarray(), result_naive, rtol=1e-5, atol=1e-8):
    print("\nResults are matching !")
else:
    print("Results do NOT match );")

egraph.pop()