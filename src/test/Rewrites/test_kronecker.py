import unittest

from Rewrites.MatrixSort import Matrix
from Rewrites.EGraph import egraph

class TestMatrixKronecker(unittest.TestCase):
    def test_kronecker_left_distributivity(self):
        egraph.push()

        x = Matrix(2, 3, 0.5)
        y = Matrix(4, 5, 0.5)
        z = Matrix(4, 5, 0.5)

        left_side = egraph.let("left", x.kron(y+z))
        right_side = egraph.let("right", (x.kron(y)) + (x.kron(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_right_distributivity(self):
        egraph.push()

        x = Matrix(2, 3, 0.5)
        y = Matrix(4, 5, 0.5)
        z = Matrix(4, 5, 0.5)

        left_side = egraph.let("left", (y+z).kron(x))
        right_side = egraph.let("right", (y.kron(x)) + (z.kron(x)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_associativity(self):
        egraph.push()

        x = Matrix(2, 3, 0.5)
        y = Matrix(4, 5, 0.5)
        z = Matrix(6, 7, 0.5)

        left_side = egraph.let("left", (x.kron(y)).kron(z))
        right_side = egraph.let("right", x.kron(y.kron(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_mixed_product(self):
        egraph.push()

        w = Matrix(2, 3, 0.0)
        x = Matrix(4, 5, 0.0)
        y = Matrix(3, 2, 0.0)
        z = Matrix(5, 4, 0.0)

        left_side = egraph.let("left", (w.kron(x)) @ (y.kron(z)))
        right_side = egraph.let("right", (w @ y).kron(x @ z))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_hadamard_distributivity(self):
        egraph.push()

        w = Matrix(2, 3, 0.5)
        x = Matrix(4, 5, 0.5)
        y = Matrix(2, 3, 0.5)
        z = Matrix(4, 5, 0.5)

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_inverse(self):
        egraph.push()

        w = Matrix(3, 3, 0.5)
        x = Matrix(2, 2, 0.5)

        left_side = egraph.let("left", w.kron(x).mat_inv())
        right_side = egraph.let("right", w.mat_inv().kron(x.mat_inv()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_transpose_distributivity(self):
        egraph.push()
        w = Matrix(3, 3, 0.5)
        x = Matrix(2, 2, 0.5)

        left_side = egraph.let("left", w.kron(x).mat_trans())
        right_side = egraph.let("right", w.mat_trans().kron(x.mat_trans()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_sparse_kron(self):
        egraph.push()
        w = Matrix(3, 3, 0.8)
        x = Matrix(8, 10, 0.9)

        left = egraph.let("left", w.kron(x))
        right = egraph.let("right", w.to_CSR().kron_sparse(x.to_CSR()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left == right))

        egraph.pop()

    def test_sparse_kron_norewrite(self):
        egraph.push()
        w = Matrix(3, 3, 0.3)
        x = Matrix(8, 10, 0.9)

        left = egraph.let("left", w.kron(x))
        right = egraph.let("right", w.to_CSR().kron_sparse(x.to_CSR()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left != right))

        egraph.pop()