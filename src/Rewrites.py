#from __future__ import annotations
#from collections.abc import Iterable
#from egglog import *
#from enum import Enum

#SPARSITY_THRESHOLD = 0.6

#class StorageFormat(Enum):
    #NATIVE = i64(1)
    #CSC = i64(2)
    #CSR = i64(3)

#class Vector(Expr):
    #def __init__(self, len: i64Like) -> None: ...

    #@property
    #def len(self) -> i64: ...

#class Matrix(Expr):
    #def __init__(self, rows: i64Like, cols: i64Like, sparsity: f64Like, storage: i64Like) -> None: ...
    #def mat_inv(self) -> Matrix: ...
    #def mat_mpinv(self) -> Matrix: ...
    #def mat_trans(self) -> Matrix: ...
    #def __matmul__(self, other: Matrix) -> Matrix: ...
    #def mat_add(self, other: Matrix) -> Matrix: ...
    #def kron(self, other: Matrix) -> Matrix: ...
    #def krao(self, other: Matrix) -> Matrix: ...
    #def hdmr(self, other: Matrix) -> Matrix: ...
    #def mat_concat(self, other: Matrix) -> Matrix: ...
    #def mat_vec_mul(self, other: Vector) -> Vector: ...
    #def spmv(self, other: Vector) -> Vector: ...
    #def to_csr(self) -> Matrix: ...
    #def to_csc(self) -> Matrix: ...

    #@property
    #def row(self) -> i64: ...
    #@property
    #def col(self) -> i64: ...
    #@property
    #def sparsity(self) -> f64: ...
    #@method(merge=lambda old, new: new)
    #@property
    #def storage(self) -> i64: ...

#egraph = EGraph()

#@egraph.register
#def _matrix_self(x: Matrix, y: Matrix, r: i64, c: i64, s: f64, st: i64) -> Iterable[RewriteOrRule]:
    #yield rule(
        #x == y.mat_trans(),
        #r == y.row,
        #c == y.col,
        #s == y.sparsity,
    #).then(
        #set_(x.row).to(c),
        #set_(x.col).to(r),
        #set_(x.sparsity).to(s),
    #)
    #yield rule(
        #x == y.mat_trans(),
        #r == y.row,
        #c == y.col,
        #s == y.sparsity,
    #).then(set_cost(y.mat_trans(), r*c))

    #yield rule(
        #x == y.mat_inv(),
        #y.row == y.col,
        #r == y.row,
        #c == y.col,
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(0.3)
    #)
    #yield rule(
        #x == y.mat_inv(),
        #r == y.row,
        #c == y.col,
    #).then(set_cost(y.mat_inv(), (2*c) * r))

#@egraph.register
#def _matrix_matrix(w: Matrix, x: Matrix, y: Matrix, z: Matrix, r: i64, c: i64, m: i64, s: f64, st: i64) -> Iterable[RewriteOrRule]:
    #yield rule(
        #x == Matrix(r, c, s, st)
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s),
        #set_(x.storage).to(st),
    #)

    #yield rule(
        #x == (y @ z),
        #y.col == z.row,
        #y.sparsity < SPARSITY_THRESHOLD,
        #z.sparsity < SPARSITY_THRESHOLD,
        #r == y.row,
        #c == z.col,
        #s == y.sparsity * z.sparsity,
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s)
    #)
    #yield rule(
        #y @ z,
        #r == y.row,
        #m == y.col,
        #c == z.col,
    #).then(set_cost(y @ z, r * m * c))

    #yield birewrite(x @ (y @ z)).to((x @ y) @ z)


    #yield rule(
        #x == y.kron(z),
        #y.sparsity < SPARSITY_THRESHOLD,
        #z.sparsity < SPARSITY_THRESHOLD,
        #y.storage == StorageFormat.NATIVE.value,
        #z.storage == StorageFormat.NATIVE.value,
        #r == y.row * z.row,
        #c == y.col * z.col,
        #s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s),
    #)
    #yield rule(
        #x == y.kron(z),
        #r == y.row * z.row,
        #c == y.col * z.col,
    #).then(set_cost(y.kron(z), r * c))
    #yield birewrite(x.kron(y.mat_add(z))).to((x.kron(y)).mat_add(x.kron(z)))
    #yield birewrite(y.mat_add(z).kron(x)).to((y.kron(x)).mat_add(z.kron(x)))
    #yield birewrite((x.kron(y)).kron(z)).to(x.kron((y.kron(z))))
    #yield birewrite((w.kron(x)) @ (y.kron(z))).to((w@y).kron(x@z))
    #yield birewrite((w.kron(x)).hdmr((y.kron(z)))
        #).to((w.hdmr(y)).kron((x.hdmr(z)))
        #, w.row == y.row
        #, w.col == y.col
        #, x.row == z.row
        #, x.col == z.col)
    #yield birewrite((w.kron(x)).mat_inv()).to(w.mat_inv().kron(x.mat_inv()))
    #yield birewrite(w.kron(x).mat_trans()).to(w.mat_trans().kron(x.mat_trans()))
    ##TO ADD: concatenation (cf. wiki 4. Commutator Property)
    ##TO ADD: mp distrib (wiki 6. inverse of kron prod)


    #yield rule(
        #x == y.krao(z),
        #y.col == z.col,
        #y.sparsity < SPARSITY_THRESHOLD,
        #z.sparsity < SPARSITY_THRESHOLD,
        #r == y.row * z.row,
        #c == y.col,
        #s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s),
    #)
    #yield rule(
        #x == y.krao(z),
        #r == y.row * z.row,
        #c == y.col,
    #).then(set_cost(y.krao(z), r * c))
    #yield birewrite(w.krao(x).mat_trans() @ w.krao(x)).to((w.mat_trans()@w).hdmr(x.mat_trans()@x))
    #yield birewrite((w.kron(x)) @ (y.krao(z))).to((w@y).krao(x@z))

    #yield rule(
        #x == y.hdmr(z),
        #y.sparsity < SPARSITY_THRESHOLD,
        #z.sparsity < SPARSITY_THRESHOLD,
        #y.col == z.col,
        #y.row == z.row,
        #r == y.row,
        #c == y.col,
        #s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s),
    #)
    #yield rule(
        #x == y.hdmr(z),
        #r == y.row,
        #c == y.col,
    #).then(set_cost(y.hdmr(z), r * c * 2))
    #yield birewrite(w.hdmr(x)).to(x.hdmr(w))
    #yield birewrite(w.hdmr(x.hdmr(y))).to((w.hdmr(x)).hdmr(y))
    #yield birewrite(w.hdmr(x.mat_add(y))).to((w.hdmr(x)).mat_add(w.hdmr(y)))


    #yield rule(
        #x == y.mat_add(z),
        #y.row == z.row,
        #y.col == z.col,
        #y.sparsity < SPARSITY_THRESHOLD,
        #z.sparsity < SPARSITY_THRESHOLD,
        #r == y.row,
        #c == y.col,
        #s == 1.0 - (1.0-y.sparsity)*(1.0-z.sparsity),
    #).then(
        #set_(x.row).to(r),
        #set_(x.col).to(c),
        #set_(x.sparsity).to(s),
    #)
    #yield rule(
        #x == y.mat_add(z),
        #r == y.row,
        #c == y.col,
    #).then(set_cost(y.mat_add(z), r*c*2))

#@egraph.register
#def _matrix_vector(m: Matrix, x: Vector, y: Vector, r: i64, c: i64, l: i64, s: f64, st: i64) -> Iterable[RewriteOrRule]:
    #yield rule(x == Vector(l)).then(set_(x.len).to(l))

    #yield rule(
        #y == (m.mat_vec_mul(x)),
        #s == m.sparsity,
        #s < SPARSITY_THRESHOLD,
        #st == m.storage,
        #st == StorageFormat.NATIVE.value,
        #x.len == m.col,
        #l == m.row,
    #).then(set_(y.len).to(l))
    #yield rule(
        #m.mat_vec_mul(x),
        #r == m.row,
        #c == m.col,
    #).then(set_cost(m.mat_vec_mul(x), r*c))

    #yield rule(
        #y == (m.spmv(x)),
        #s == m.sparsity,
        #s >= SPARSITY_THRESHOLD,
        #st == m.storage,
        #st != StorageFormat.NATIVE.value,
        #x.len == m.col,
        #l == m.row,
    #).then(set_(y.len).to(l))
    #yield rule(
        #m.spmv(x),
        #r == m.row,
        #c == m.col,
    #).then(set_cost(m.spmv(x), r*c/2))

    #yield rewrite(m.mat_vec_mul(x)).to(
        #m.spmv(x),
        #s == m.sparsity,
        #s >= SPARSITY_THRESHOLD,
        #st == m.storage,
        #st != StorageFormat.NATIVE.value,
    #)

#@egraph.register
#def _matrix_format(a: Matrix, b: Matrix, r: i64, c: i64, s: f64, st: i64) -> Iterable[RewriteOrRule]:
    #yield rewrite(a.to_csc()).to(
        #Matrix(r, c, s, StorageFormat.CSC.value),
        #a == Matrix(r, c, s, st),
        #s >= SPARSITY_THRESHOLD,
    #)
    #yield rule(
        #a.to_csc(),
        #a == Matrix(r, c, s, st),
        #s >= SPARSITY_THRESHOLD,
    #).then(set_cost(a.to_csc(), r * c))

    #yield rewrite(a.to_csr()).to(
        #Matrix(r, c, s, StorageFormat.CSR.value),
        #a == Matrix(r, c, s, st),
        #s >= SPARSITY_THRESHOLD,
    #)
    #yield rule(
        #a.to_csr(),
        #a == Matrix(r, c, s, st),
        #s >= SPARSITY_THRESHOLD,
    #).then(set_cost(a.to_csr(), r * c))