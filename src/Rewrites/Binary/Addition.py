from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix

@egraph.register
def _matrix_addition(w: Matrix, x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y+z,
        y.row == z.row,
        y.col == z.col,
        y.sparsity < SPARSITY_THRESHOLD,
        z.sparsity < SPARSITY_THRESHOLD,
        r == y.row,
        c == y.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )
    yield rule(
        x == y+z,
        r == y.row,
        c == y.col,
    ).then(set_cost(y+z, r*c))

    yield rewrite(x @ (y+z)).to((x @ y) + (x @ z))
    yield rewrite((y+z) @ x).to((y @ x) + (z @ x))