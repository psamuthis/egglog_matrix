from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix

@egraph.register
def _matrix_khatrirao(w: Matrix, x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.krao(z),
        y.col == z.col,
        y.sparsity < SPARSITY_THRESHOLD,
        z.sparsity < SPARSITY_THRESHOLD,
        r == y.row * z.row,
        c == y.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )
    yield rule(
        x == y.krao(z),
        r == y.row * z.row,
        c == y.col,
    ).then(set_cost(y.krao(z), r * c))

    yield birewrite(w.krao(x).mat_trans() @ w.krao(x)).to((w.mat_trans()@w).hdmr(x.mat_trans()@x))
    yield birewrite((w.kron(x)) @ (y.krao(z))).to((w@y).krao(x@z))

    yield rule(
        x == (y.krao_sparse(z)),
        y.col == z.col,
        y.sparsity >= SPARSITY_THRESHOLD,
        z.sparsity >= SPARSITY_THRESHOLD,
        r == y.row * z.row,
        c == y.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )
    yield rule(
        y.krao_sparse(z),
        r == y.row * z.row,
        c == y.col,
    ).then(set_cost(y.krao(z), r*c/2))

    yield rewrite(y.krao(z)).to(
        y.to_CSR().krao_sparse(z.to_CSC()),
        y.sparsity >= SPARSITY_THRESHOLD,
        z.sparsity >= SPARSITY_THRESHOLD,
    )