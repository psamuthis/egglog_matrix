from __future__ import annotations
from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import Matrix

@egraph.register
def _matrix_transpose(x: Matrix, y: Matrix, r: i64, c: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.mat_trans(),
        r == y.row,
        c == y.col,
        s == y.sparsity,
    ).then(
        set_(x.row).to(c),
        set_(x.col).to(r),
        set_(x.sparsity).to(s),
    )
    yield rule(
        x == y.mat_trans(),
        r == y.row,
        c == y.col,
        s == y.sparsity,
    ).then(set_cost(y.mat_trans(), r*c))