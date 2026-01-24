from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix

@egraph.register
def _matrix_hadamard(w: Matrix, x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.hdmr(z),
        y.sparsity < SPARSITY_THRESHOLD,
        z.sparsity < SPARSITY_THRESHOLD,
        y.col == z.col,
        y.row == z.row,
        r == y.row,
        c == y.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )
    yield rule(
        y.hdmr(z),
        r == y.row,
        c == y.col,
    ).then(set_cost(y.hdmr(z), r * c * 2))
    yield birewrite(w.hdmr(x)).to(x.hdmr(w))
    yield birewrite(w.hdmr(x.hdmr(y))).to((w.hdmr(x)).hdmr(y))
    yield birewrite(w.hdmr(x+y)).to((w.hdmr(x)) + (w.hdmr(y)))

    yield rule(
        x == y.hdmr_sparse(z),
        y.col == z.col,
        y.row == z.row,
        y.sparsity >= SPARSITY_THRESHOLD,
        z.sparsity >= SPARSITY_THRESHOLD,
        r == y.row,
        c == y.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )

    yield rule(
        y.hdmr_sparse(z),
        r == y.row,
        c == y.col,
    ).then(set_cost(y.hdmr_sparse(z), (r * c)))

    yield rewrite(y.hdmr(z)).to(
        y.to_CSR().hdmr_sparse(z.to_CSR()),
        y.sparsity >= SPARSITY_THRESHOLD,
        z.sparsity >= SPARSITY_THRESHOLD,
    )