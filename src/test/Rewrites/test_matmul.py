import unittest

from Rewrites.MatrixSort import Matrix
from Rewrites.EGraph import egraph

class TestMatrixMultiplication(unittest.TestCase):
    def test_matmul_associativity(self):
        egraph.push()

        input = egraph.let(
            "associativity-example",
            (Matrix(64, 8, 0.0) @
            Matrix(8, 256, 0.0)) @
            Matrix(256, 2, 0.0)
        )
        expected = egraph.let(
            "expected",
            Matrix(64, 8, 0.0) @
            (Matrix(8, 256, 0.0) @
            Matrix(256, 2, 0.0))
        )

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))

        egraph.pop()

    def test_matmul_to_sparse(self):
        egraph.push()
        
        input = egraph.let("input", Matrix(256, 2, 0.8) @ Matrix(2, 256, 0.8))
        expected = egraph.let("expected", Matrix(256, 2, 0.8).to_CSR().matmul_sparse(Matrix(2, 256, 0.8).to_CSC()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        
        egraph.pop()

    def test_matmul_to_sparse_no_rewrite(self):
        egraph.push()

        input = egraph.let("input", Matrix(256, 2, 0.3) @ Matrix(2, 256, 0.4))
        avoid = egraph.let("avoid", Matrix(256, 2, 0.8).to_CSR().matmul_sparse(Matrix(2, 256, 0.8).to_CSC()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input != avoid))

        egraph.pop()