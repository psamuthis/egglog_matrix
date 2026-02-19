import numpy as np
from egglog import *

from rewrites.EGraph import egraph
from rewrites import Matrix
from compiler.compiler import compile

egraph.push()

matrices = {
    'a': Matrix('a', 64, 8, 0.0),
    'b': Matrix('b', 8, 256, 0.1),
    'c': Matrix('c', 256, 2, 0.2),
}

data = {
    'a': np.random.randn(64, 8),
    'b': np.random.randn(8, 256),
    'c': np.random.randn(256, 2),
}

expr = egraph.let("expr", (matrices['a']@matrices['b'])@matrices['c'])
result = compile(expr, egraph, data, DEBUG=True)
print(f'Resulting matrix has shape: {result.shape}')
print(f'First five resulting rows:\n{result[:5, :]}')

egraph.pop()