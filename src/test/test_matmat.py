import unittest
from matrix import egraph, Matrix, Vector, SPARSITY_THRESHOLD, StorageFormat

class TestMatrixMatrix(unittest.TestCase):
    def test_matmul(self):
        egraph.push()
        input = egraph.let(
            "associativity-example",
            (Matrix(64, 8, 0.0, StorageFormat.NATIVE.value) @
            Matrix(8, 256, 0.0, StorageFormat.NATIVE.value)) @
            Matrix(256, 2, 0.0, StorageFormat.NATIVE.value)
        )
        expected = egraph.let(
            "expected",
            Matrix(64, 8, 0.0, StorageFormat.NATIVE.value) @
            (Matrix(8, 256, 0.0, StorageFormat.NATIVE.value) @
            Matrix(256, 2, 0.0, StorageFormat.NATIVE.value))
        )

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()