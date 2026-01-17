import unittest
from Rewrites import egraph, Matrix, Vector, SPARSITY_THRESHOLD, StorageFormat

class TestMatrixVector(unittest.TestCase):

    def test_spmv_rewrite(self):
        egraph.push()
        common_dimension = 29
        matrix_sparse = egraph.let("matrix_sparse", Matrix(64, common_dimension, 0.7, StorageFormat.CSC.value))
        vector = egraph.let("vector", Vector(common_dimension))
        input = egraph.let("input", matrix_sparse.mat_vec_mul(vector))
        expected = egraph.let("expected", matrix_sparse.spmv(vector))

        egraph.saturate(visualize=False)

        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_no_rewrite_for_dense(self):
        egraph.push()
        matrix_dense = egraph.let("matrix_dense", Matrix(64, 29, 0.5, StorageFormat.NATIVE.value))
        vector = egraph.let("vector", Vector(29))
        input = egraph.let("input", matrix_dense.mat_vec_mul(vector))
        avoid = egraph.let("avoid", matrix_dense.spmv(vector))

        egraph.saturate(visualize=False)

        self.assertFalse(egraph.check_bool(input == avoid))
        egraph.pop()