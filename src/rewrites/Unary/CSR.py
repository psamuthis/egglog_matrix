from __future__ import annotations
from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import Matrix
from ..MatrixSort import SPARSITY_THRESHOLD

@egraph.register
def _matrix_CSR(x: Matrix, y: Matrix, r: i64, c: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.to_CSR(),
        y.sparsity >= SPARSITY_THRESHOLD,
        y.sparsity < 1,
        r == y.row,
        c == y.col,
        s == y.sparsity,
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )

    yield rule(
        y.to_CSR(),
        r == y.row,
        c == y.col,
    ).then(set_cost(y.to_CSR(), r*c))