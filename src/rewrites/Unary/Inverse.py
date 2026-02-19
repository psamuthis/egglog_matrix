from __future__ import annotations
from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import Matrix

@egraph.register
def _matrix_inverse(x: Matrix, y: Matrix, r: i64, c: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.mat_inv(),
        y.row == y.col,
        r == y.row,
        c == y.col,
        s == y.sparsity,
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s**0.25)
    )
    yield rule(
        x == y.mat_inv(),
        r == y.row,
        c == y.col,
    ).then(set_cost(y.mat_inv(), (2*c) * r))