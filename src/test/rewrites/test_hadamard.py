import unittest

from rewrites.MatrixSort import Matrix
from rewrites.EGraph import egraph

class TestMatrixHadamard(unittest.TestCase):
    def test_hadamard_commutativity(self):
        """Test: w.hdmr(x) -> x.hdmr(w)"""
        egraph.push()

        w = Matrix("w", 3, 4, 0.5)
        x = Matrix("x", 3, 4, 0.5)

        left_side = egraph.let("left", w.hdmr(x))
        right_side = egraph.let("right", x.hdmr(w))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hadamard_associativity(self):
        """Test: w.hdmr(x.hdmr(y)) -> (w.hdmr(x)).hdmr(y)"""
        egraph.push()

        w = Matrix("w", 3, 4, 0.5)
        x = Matrix("x", 3, 4, 0.5)
        y = Matrix("y", 3, 4, 0.5)

        left_side = egraph.let("left", w.hdmr(x.hdmr(y)))
        right_side = egraph.let("right", (w.hdmr(x)).hdmr(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hadamard_distributivity_over_addition(self):
        egraph.push()

        w = Matrix("w", 3, 4, 0.5)
        x = Matrix("x", 3, 4, 0.5)
        y = Matrix("y", 3, 4, 0.5)

        left_side = egraph.let("left", w.hdmr(x+y))
        right_side = egraph.let("right", (w.hdmr(x)) + (w.hdmr(y)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hdmr_kron_hdmr(self):
        egraph.push()
        w = Matrix("w", 2, 3, 0.5)
        x = Matrix("x", 4, 5, 0.5)
        y = Matrix("y", 2, 3, 0.5)  # same dims as w
        z = Matrix("z", 4, 5, 0.5)  # same dims as x

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_hadamard_distributivity_fails_wrong_dims(self):
        egraph.push()

        w = Matrix("w", 2, 3, 0.5)
        x = Matrix("x", 4, 5, 0.5)
        y = Matrix("y", 3, 2, 0.5)
        z = Matrix("z", 4, 5, 0.5)

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertFalse(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hdmr_to_sparse(self):
        egraph.push()

        x = Matrix("x", 24, 48, 0.8)
        y = Matrix("y", 24, 48, 0.7)
        input = egraph.let("input", x.hdmr(y))
        expected = egraph.let("expected", x.hdmr_sparse(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))

        egraph.pop()

    def test_hdmr_to_sparse_no_rewrite(self):
        egraph.push()

        x = Matrix("x", 24, 48, 0.3)
        y = Matrix("y", 24, 48, 0.2)
        input = egraph.let("input", x.hdmr(y))
        avoid = egraph.let("avoid", x.hdmr_sparse(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input != avoid))

        egraph.pop()