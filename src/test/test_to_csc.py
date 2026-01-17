import unittest
from Rewrites import egraph, Matrix, Vector, SPARSITY_THRESHOLD, StorageFormat

class TestMatrixToCSC(unittest.TestCase):
    def test_native_to_csc_rewrite(self):
        egraph.push()
        hidden_sparse_matrix = egraph.let("matrix_sparse", Matrix(64, 29, 0.7, StorageFormat.NATIVE.value))
        input = egraph.let("input", hidden_sparse_matrix.to_csc())
        expected = egraph.let("expected", Matrix(64, 29, 0.7, StorageFormat.CSC.value))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_csr_to_csc_rewrite(self):
        egraph.push()
        csr_matrix = egraph.let("csr-matrix", Matrix(64, 29, 0.7, StorageFormat.CSR.value))
        input = egraph.let("input", csr_matrix.to_csc())
        expected = egraph.let("expected", Matrix(64, 29, 0.7, StorageFormat.CSC.value))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_csc_to_csc_rewrite(self):
        egraph.push()
        csc_matrix = egraph.let("csc-matrix", Matrix(64, 29, 0.7, StorageFormat.CSC.value))
        input = egraph.let("input", csc_matrix.to_csc())
        expected = egraph.let("expected", Matrix(64, 29, 0.7, StorageFormat.CSC.value))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_no_to_csc_for_dense(self):
        egraph.push()
        dense_matrix = egraph.let("dense-matrix", Matrix(64, 29, 0.4, StorageFormat.NATIVE.value))
        input = egraph.let("input", dense_matrix.to_csr())
        avoid = egraph.let("expected", Matrix(64, 29, 0.4, StorageFormat.CSC.value))

        egraph.saturate(visualize=False)
        print(egraph.extract(input))
        self.assertTrue(egraph.check_bool(input != avoid))
        egraph.pop()