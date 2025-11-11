from __future__ import annotations
from collections.abc import Iterable
from egglog import *

SPARSITY_THRESHOLD = 0.6

class Vector(Expr):
    def __init__(self, len: i64Like) -> None: ...

    @property
    def len(self) -> i64: ...

class Matrix(Expr):
    def __init__(self, rows: i64Like, cols: i64Like, sparsity: f64Like, storage: StringLike) -> None: ...
    def __matmul__(self, other: Matrix) -> Matrix: ...
    def mat_vec_mul(self, other: Vector) -> Vector: ...
    def spmv(self, other: Vector) -> Vector: ...

    @property
    def row(self) -> i64: ...
    @property
    def col(self) -> i64: ...
    @property
    def sparsity(self) -> f64: ...
    @property
    def storage(self) -> StringLike: ...

egraph = EGraph()

@egraph.register
def _matrix_matrix(x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64, st: String) -> Iterable[RewriteOrRule]:
    yield rule(
        x == Matrix(r, c, s, st)
    ).then(set_(x.row).to(r), set_(x.col).to(c), set_(x.sparsity).to(s), set_(x.storage).to(st))
    yield rule(
        x == (y @ z),
        y.col == z.row,
        r == y.row,
        c == z.col,
    ).then(set_(x.row).to(r), set_(x.col).to(c))

    yield rule(
        y @ z,
        r == y.row,
        m == y.col,
        c == z.col,
    ).then(set_cost(y @ z, r * m * c))

    yield birewrite(x @ (y @ z)).to((x @ y) @ z)

@egraph.register
def _matrix_vector(m: Matrix, x: Vector, y: Vector, r: i64, c: i64, l: i64, s: f64, st: String) -> Iterable[RewriteOrRule]:
    yield rule(x == Vector(l)).then(set_(x.len).to(l))

    yield rule(
        y == (m.mat_vec_mul(x)),
        s == m.sparsity,
        s < SPARSITY_THRESHOLD,
        x.len == m.col,
        l == m.row
    ).then(set_(y.len).to(l))
    yield rule(
        m.spmv(x),
        r == m.row,
        c == m.col,
    ).then(set_cost(m.mat_vec_mul(x), r*c))

    yield rule(
        y == (m.spmv(x)),
        s == m.sparsity,
        s >= SPARSITY_THRESHOLD,
        x.len == m.col,
        l == m.row,
    ).then(set_(y.len).to(l))
    yield rule(
        m.spmv(x),
        r == m.row,
        c == m.col,
        s == m.sparsity,
    ).then(set_cost(m.spmv(x), r*c/2))

    yield rewrite(m.mat_vec_mul(x)).to(
        m.spmv(x),
        s == m.sparsity,
        s >= SPARSITY_THRESHOLD,
    )

matrix_sparse = egraph.let("matrix_sparse", Matrix(64, 29, 0.7, "sparse"))
matrix_dense = egraph.let("matrix_dense", Matrix(64, 29, 0.3, "dense"))
vector1 = egraph.let("vector1", Vector(29))

sparse_expr = egraph.let("sparse_expr", matrix_sparse.mat_vec_mul(vector1))
sparse_expected = egraph.let("sparse_expected", matrix_sparse.spmv(vector1))
dense_expr = egraph.let("dense_expr", matrix_dense.mat_vec_mul(vector1))
dense_expected = egraph.let("dense_expected", matrix_dense.mat_vec_mul(vector1))

egraph.saturate()

egraph.check(sparse_expr == sparse_expected)
egraph.check(dense_expr == dense_expected)
#print(egraph.extract(sparse_expr))