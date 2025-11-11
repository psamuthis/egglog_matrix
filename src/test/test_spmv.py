import unittest
from matrix import egraph, Matrix, Vector, SPARSITY_THRESHOLD

class TestMatrixVector(unittest.TestCase):

    def test_spmv_rewrite(self):
        egraph.push()
        matrix_sparse = egraph.let("matrix_sparse", Matrix(64, 29, 0.7, "sparse"))
        vector = egraph.let("vector", Vector(29))
        input = egraph.let("input", matrix_sparse.mat_vec_mul(vector))
        expected = egraph.let("expected", matrix_sparse.spmv(vector))

        egraph.saturate()

        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_no_rewrite_for_dense(self):
        egraph.push()
        matrix_dense = egraph.let("matrix_dense", Matrix(64, 29, 0.5, "dense"))
        vector = egraph.let("vector", Vector(29))
        input = egraph.let("input", matrix_dense.mat_vec_mul(vector))
        avoid = egraph.let("avoid", matrix_dense.spmv(vector))

        egraph.saturate()

        self.assertFalse(egraph.check_bool(input == avoid))
        egraph.pop()