from egglog import *
from collections.abc import Iterable

from ..EGraph import egraph
from ..MatrixSort import SPARSITY_THRESHOLD
from ..MatrixSort import Matrix

@egraph.register
def _matrix_kronecker(w: Matrix, x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64) -> Iterable[RewriteOrRule]:
    yield rule(
        x == y.kron(z),
        y.sparsity < SPARSITY_THRESHOLD,
        z.sparsity < SPARSITY_THRESHOLD,
        r == y.row * z.row,
        c == y.col * z.col,
        s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    ).then(
        set_(x.row).to(r),
        set_(x.col).to(c),
        set_(x.sparsity).to(s),
    )
    yield rule(
        x == y.kron(z),
        r == y.row * z.row,
        c == y.col * z.col,
    ).then(set_cost(y.kron(z), r * c))

    yield rule(
        x == y.kron_sparse(z),
        r == y.row * z.row,
        c == y.col * z.col,
    ).then(set_cost(y.kron_sparse(z), r*c/2))

    yield birewrite(x.kron(y+z)).to((x.kron(y)) + (x.kron(z)))
    yield birewrite((y+z).kron(x)).to((y.kron(x)) + (z.kron(x)))
    yield birewrite((x.kron(y)).kron(z)).to(x.kron((y.kron(z))))
    yield birewrite((w.kron(x)) @ (y.kron(z))).to((w@y).kron(x@z))
    yield birewrite((w.kron(x)).hdmr((y.kron(z)))
        ).to((w.hdmr(y)).kron((x.hdmr(z)))
        , w.row == y.row
        , w.col == y.col
        , x.row == z.row
        , x.col == z.col)
    yield birewrite((w.kron(x)).mat_inv()).to(w.mat_inv().kron(x.mat_inv()))
    yield birewrite(w.kron(x).mat_trans()).to(w.mat_trans().kron(x.mat_trans()))
    #TO ADD: concatenation (cf. wiki 4. Commutator Property)
    #TO ADD: mp distrib (wiki 6. inverse of kron prod)

    yield birewrite(x.kron(y)).to(
        x.to_CSR().kron_sparse(y.to_CSR()),
        x.sparsity >= SPARSITY_THRESHOLD,
        y.sparsity >= SPARSITY_THRESHOLD,
    )