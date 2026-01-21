from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix
from ..VectorSort import Vector

@egraph.register
def _matrix_vector(m: Matrix, x: Vector, y: Vector, r: i64, c: i64, l: i64, s: f64) -> Iterable[RewriteOrRule]:

    yield rule(
        y == (m.mat_vec_mul(x)),
        s == m.sparsity,
        s < SPARSITY_THRESHOLD,
        x.len == m.col,
        l == m.row,
    ).then(set_(y.len).to(l))
    yield rule(
        m.mat_vec_mul(x),
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
    ).then(set_cost(m.spmv(x), r*c/2))

    yield rewrite(m.mat_vec_mul(x)).to(
        m.spmv(x),
        s == m.sparsity,
        s >= SPARSITY_THRESHOLD,
    )