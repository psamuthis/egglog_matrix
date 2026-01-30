import unittest

from Rewrites.MatrixSort import Matrix
from Rewrites.EGraph import egraph

class TestMatrixKhatriRao(unittest.TestCase):
    def test_krao_transpose_times_krao(self):
        egraph.push()
        w = Matrix(3, 3, 0.5)
        x = Matrix(2, 2, 0.5)

        left_side = egraph.let("left", w.krao(x).mat_trans() @ w.krao(x))
        right_side = egraph.let("right", (w.mat_trans()@w).hdmr(x.mat_trans()@x))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kron_times_krao_distributivity(self):
        egraph.push()
        w = Matrix(3, 3, 0.5)
        x = Matrix(2, 2, 0.5)
        y = Matrix(4, 2, 0.5)
        z = Matrix(3, 2, 0.5)

        left_side = egraph.let("left", (w.kron(x)) @ (y.krao(z)))
        right_side = egraph.let("right", (w@y).krao(x@z))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_krao_to_sparse(self):
        egraph.push()

        x = Matrix(43, 6, 0.8)
        y = Matrix(47, 6, 0.8)

        input = egraph.let("input", x.krao(y))
        expected = egraph.let("expected", x.to_CSR().krao_sparse(y.to_CSC()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))

        egraph.pop()

    def test_krao_to_sparse_no_rewrite(self):
        egraph.push()

        x = Matrix(43, 6, 0.3)
        y = Matrix(47, 6, 0.8)

        input = egraph.let("input", x.krao(y))
        avoid = egraph.let("avoid", x.to_CSR().krao_sparse(y.to_CSC()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input != avoid))

        egraph.pop()