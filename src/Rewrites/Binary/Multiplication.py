from egglog import *
from collections.abc import Iterable
import unittest

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix

@egraph.register
def _matrix_multiplication(x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(x == Matrix(r, c, s)).then(set_(x.row).to(r), set_(x.col).to(c), set_(x.sparsity).to(s))
    yield rule(
        x == (y @ z),
        y.col == z.row,
        y.sparsity < SPARSITY_THRESHOLD,
        z.sparsity < SPARSITY_THRESHOLD,
        r == y.row,
        c == z.col,
        s == y.sparsity * z.sparsity,
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s)
    )
    yield rule(
        y @ z,
        r == y.row,
        m == y.col,
        c == z.col,
    ).then(set_cost(y @ z, r * m * c))

    yield birewrite(x @ (y @ z)).to((x @ y) @ z)

    yield rule(
        x == y.matmul_sparse(z),
        y.col == z.row,
        y.sparsity >= SPARSITY_THRESHOLD,
        z.sparsity >= SPARSITY_THRESHOLD,
        r == y.row,
        c == z.col,
        s == y.sparsity * z.sparsity,
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )

    yield rule(
        y.matmul_sparse(z),
        r == y.row,
        c == y.col,
        m == z.col,
    ).then(set_cost(y.matmul_sparse(z), (r*m*c)/2))

    yield rewrite(x @ y).to(
        x.to_CSR().matmul_sparse(y.to_CSC()),
        x.sparsity >= SPARSITY_THRESHOLD,
        y.sparsity >= SPARSITY_THRESHOLD,
    )