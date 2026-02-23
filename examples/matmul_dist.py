import numpy as np
from egglog import *
import time

from rewrites.EGraph import egraph
from rewrites import Matrix
from compiler.compiler import compile
from compiler.compiler import comp_timer
from utils.sparse_randn import sparse_randn

egraph.push()

data = {
    'a': sparse_randn((64000, 800), 0.2),
    'b': sparse_randn((800, 2560), 0.3),
    'c': sparse_randn((2560, 2000), 0.1),
}

matrices = {
    'a': Matrix('a', 64000, 800, 0.2),
    'b': Matrix('b', 800, 2560, 0.3),
    'c': Matrix('c', 2560, 2000, 0.1),
}

expr = egraph.let("expr", (matrices['a']@matrices['b'])@matrices['c'])
result = compile(expr, egraph, data, DEBUG=True)
comp_timer.report()

print("\nPerforming naive operation...")
start = time.perf_counter()
result_naive = (data['a'] @ data['b']) @ data['c']
end = time.perf_counter()
print(f'Naive version ran in: {end-start:.6f}')

if np.allclose(result, result_naive, rtol=1e-5, atol=1e-8):
    print("\nResults are matching !")
else:
    print("Results do NOT match );")

egraph.pop()