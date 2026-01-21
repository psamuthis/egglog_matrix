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