import unittest

from rewrites.MatrixSort import Matrix
from rewrites.EGraph import egraph

class TestMatrixKhatriRao(unittest.TestCase):
    def test_krao_transpose_times_krao(self):
        egraph.push()
        w = Matrix("w", 3, 3, 0.5)
        x = Matrix("x", 2, 2, 0.5)

        left_side = egraph.let("left", w.krao(x).mat_trans() @ w.krao(x))
        right_side = egraph.let("right", (w.mat_trans()@w).hdmr(x.mat_trans()@x))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kron_times_krao_distributivity(self):
        egraph.push()
        w = Matrix("w", 3, 3, 0.5)
        x = Matrix("x", 2, 2, 0.5)
        y = Matrix("y", 4, 2, 0.5)
        z = Matrix("z", 3, 2, 0.5)

        left_side = egraph.let("left", (w.kron(x)) @ (y.krao(z)))
        right_side = egraph.let("right", (w@y).krao(x@z))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_krao_to_sparse(self):
        egraph.push()

        x = Matrix("x", 43, 6, 0.8)
        y = Matrix("y", 47, 6, 0.8)

        input = egraph.let("input", x.krao(y))
        expected = egraph.let("expected", x.krao_sparse(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))

        egraph.pop()

    def test_krao_to_sparse_no_rewrite(self):
        egraph.push()

        x = Matrix("x", 43, 6, 0.3)
        y = Matrix("y", 47, 6, 0.8)

        input = egraph.let("input", x.krao(y))
        avoid = egraph.let("avoid", x.krao_sparse(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input != avoid))

        egraph.pop()