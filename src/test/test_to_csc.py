import unittest
from matrix import egraph, Matrix, Vector, SPARSITY_THRESHOLD, StorageFormat

class TestMatrixToCSC(unittest.TestCase):
    def test_native_to_csc_rewrite(self):
        egraph.push()
        hidden_sparse_matrix = egraph.let("matrix_sparse", Matrix(64, 29, 0.7, StorageFormat.NATIVE.value))
        input = egraph.let("input", hidden_sparse_matrix.to_csc())
        expected = egraph.let("expected", Matrix(64, 29, 0.7, StorageFormat.CSC.value))

        egraph.saturate(visualize=False)
        print(egraph.extract(input))
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()