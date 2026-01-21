import numpy as np
from egglog import *

from Rewrites.EGraph import egraph
from Rewrites import Matrix
from Compiler import compile

egraph.push()

a_data = np.random.randn(64, 8)
b_data = np.random.randn(8, 256)
c_data = np.random.randn(256, 2)

a = Matrix(64, 8, 0.0)
b = Matrix(8, 256, 0.1)
c = Matrix(256, 2, 0.2)
d = Matrix(256, 2, 0.3)
e = Matrix(256, 2, 0.4)
f = Matrix(2, 256, 0.4)

#expr = egraph.let("expr", ((a @ b) @ c).kron(d.mat_add(e)) )
expr = egraph.let("expr", (a @ b) @ c)
#expr = egraph.let("expr", a@(b@(c@(d@e))) )
#expr = egraph.let("expr", (((e@d)@c)@b)@a )
#expr = egraph.let("expr", a @ b.mat_trans())
#expr = egraph.let("expr", f @ (d + e))
#compile(expr, egraph)

print(f'Input expression: {str(egraph.extract(expr))}')
egraph.saturate(visualize=False)

output_expr = egraph.extract(expr)
print(f'Output expression: {output_expr}')


egraph.pop()
