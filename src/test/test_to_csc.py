import unittest
from matrix import egraph, Matrix, Vector, SPARSITY_THRESHOLD

class TestMatrixToCSC(unittest.TestCase):
    def test_spmv_rewrite(self):
        egraph.push()
        #hidden_sparse_matrix = egraph.let("matrix_sparse", Matrix(64, 29, 0.7, "dense"))
        #input = egraph.let("input", hidden_sparse_matrix.to_csc())
        #expected = egraph.let("expected", Matrix(64, 29, 0.7, "sparse"))

        #egraph.saturate()

        #print("coucou")
        #print(egraph.extract(input))
        #self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()