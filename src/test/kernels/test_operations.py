import pytest
import numpy as np
import scipy.sparse as sp
from rewrites.MatrixSort import Matrix
from rewrites.EGraph import egraph
from compiler.compiler import compile

def test_matmul_dense_vs_sparse():
    """Verify that matmul_sparse gives the same result as dense @."""
    # Create test data
    A_data = np.random.randn(10, 20)
    B_data = np.random.randn(20, 10)
    
    # Define in egraph
    from rewrites import Matrix
    A_expr = Matrix("test_a", 10, 20, 0.0)
    B_expr = Matrix("test_b", 20, 10, 0.8) # High sparsity to trigger sparse rules
    
    symbol_table = {"test_a": A_data, "test_b": B_data}
    
    # 1. Test Dense @ (Standard)
    expr_dense = A_expr @ B_expr
    res_dense = compile(expr_dense, egraph, symbol_table)
    
    # 2. Test Sparse Matmul (Force Sparse)
    expr_sparse = A_expr.matmul_sparse(B_expr)
    res_sparse = compile(expr_sparse, egraph, symbol_table)
    
    assert np.allclose(res_dense, res_sparse.toarray() if sp.issparse(res_sparse) else res_sparse)
    assert np.allclose(res_dense, A_data @ B_data)

def test_khatri_rao_correctness():
    """Verify the vectorized Khatri-Rao implementation."""
    A = np.array([[1, 2], 
                  [3, 4]])
    B = np.array([[5, 6], 
                  [7, 8]])
    
    # Expected: col-wise Kronecker
    # col1: kron([1,3], [5,7]) = [5, 7, 15, 21]
    # col2: kron([2,4], [6,8]) = [12, 16, 24, 32]
    expected = np.array([[5, 12],
                         [7, 16],
                         [15, 24],
                         [21, 32]])
    
    from compiler.compiler import khatri_rao_dense
    result = khatri_rao_dense(A, B)
    assert np.allclose(result, expected)

def test_hadamard_sparse():
    """Test Hadamard (element-wise) product with SciPy sparse backend."""
    A_data = sp.random(10, 10, density=0.1, format='csr')
    B_data = sp.random(10, 10, density=0.1, format='csr')
    
    from rewrites import Matrix
    A_expr = Matrix("sa", 10, 10, 0.9)
    B_expr = Matrix("sb", 10, 10, 0.9)
    
    symbol_table = {"sa": A_data, "sb": B_data}
    expr = A_expr.hdmr_sparse(B_expr)
    
    result = compile(expr, egraph, symbol_table)
    
    # SciPy multiply is the Hadamard product for sparse matrices
    expected = A_data.multiply(B_data)
    assert (result != expected).nnz == 0

@pytest.mark.parametrize("op_name", ["kron", "krao"])
def test_kronecker_variants(op_name):
    """Test Kronecker and Khatri-Rao via the compiler."""
    A_data = np.random.rand(4, 2)
    B_data = np.random.rand(4, 2)
    
    from rewrites import Matrix
    A_expr = Matrix("ka", 4, 2, 0.0)
    B_expr = Matrix("kb", 4, 2, 0.0)
    
    symbol_table = {"ka": A_data, "kb": B_data}
    
    if op_name == "kron":
        expr = A_expr.kron(B_expr)
        expected = np.kron(A_data, B_data)
    else:
        expr = A_expr.krao(B_expr)
        from compiler.compiler import khatri_rao_dense
        expected = khatri_rao_dense(A_data, B_data)
        
    result = compile(expr, egraph, symbol_table)
    assert np.allclose(result, expected)