import numpy as np
from Rewrites import Matrix, Vector, egraph, StorageFormat
from Compiler import compile

egraph.push()

a_data = np.random.randn(64, 8)
b_data = np.random.randn(8, 256)
c_data = np.random.randn(256, 2)

a = Matrix(64, 8, 0.0, StorageFormat.NATIVE.value)
b = Matrix(8, 256, 0.1, StorageFormat.NATIVE.value)
c = Matrix(256, 2, 0.2, StorageFormat.NATIVE.value)
d = Matrix(256, 2, 0.3, StorageFormat.NATIVE.value)
e = Matrix(256, 2, 0.4, StorageFormat.NATIVE.value)

#expr = egraph.let("expr", ((a @ b) @ c).kron(d.mat_add(e)) )
#expr = egraph.let("expr", a@(b@c) )
#expr = egraph.let("expr", a@(b@(c@(d@e))) )
#expr = egraph.let("expr", (((e@d)@c)@b)@a )
expr = egraph.let("expr", a @ b.mat_trans())
compile(expr, egraph)

egraph.pop()