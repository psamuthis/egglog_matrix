from egglog import Expr
from pathlib import Path
import re
import ctypes
import subprocess
import shutil
import numpy as np
import tempfile
import os
import scipy
import scipy.sparse as sp

from .parser import build_compute_graph
from .parser import BinaryMatrixOp
from .parser import UnaryMatrixOp
from .parser import ExpressionTree
from .parser import MATRIX_TOKEN
from .parser import MATRIX_RAW
from utils.time_operation import ComputationTimer

comp_timer = ComputationTimer()

def compile(expr, egraph, matrices_data=None, DEBUG=False):
    if(DEBUG):
        print(f"Rewriting original expression:\n{egraph.extract(expr)}")
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    optimized_expr = str(optimized_expr)
    if(DEBUG):
        print(f'\nExpression as been rewritten to:\n{optimized_expr}')

    if(DEBUG):
        print(f'\nBuilding compute graph...')
    compute_graph = build_compute_graph(optimized_expr)
    compute_graph = lift_unary_ops(compute_graph)
    compute_graph = unfold_op_dist_over_add(compute_graph)
    if(DEBUG):
        print(f'Resulting compute graph:')
        compute_graph.print_tree()

    return evaluate(compute_graph, matrices_data)


def evaluate(node, data_map):
    if isinstance(node, str):
        match = re.search(r'Matrix\("([^"]+)"', node)
        if match:
            matrix_id = match.group(1)
            val = data_map.get(matrix_id)
            if val is None:
                raise KeyError(f"Matrix ID '{matrix_id}' not found in data_map.")
            return val
        return data_map.get(node)

    left_val = evaluate(node.left, data_map) if node.left else None
    right_val = evaluate(node.right, data_map) if node.right else None

    if node.left is None and node.right is None:
        match = re.search(r'Matrix\("([^"]+)"', node.node)
        if match:
            matrix_id = match.group(1)
            val = data_map.get(matrix_id)
            if val is None:
                raise KeyError(f"Matrix ID '{matrix_id}' not found in data_map.")
            return val
        return data_map.get(node.node)

    def ensure_csr(val):
        return val if sp.isspmatrix_csr(val) else sp.csr_matrix(val)

    def ensure_csc(val):
        return val if sp.isspmatrix_csc(val) else sp.csc_matrix(val)

    def ensure_dense(val):
        return val.toarray() if sp.issparse(val) else val

    @comp_timer.time_operation("@")
    def matmul_op(l, r): return l @ r

    @comp_timer.time_operation("hdmr")
    def hdmr_op(l, r): return l * r

    @comp_timer.time_operation("kron")
    def kron_op(l, r): return np.kron(l, r)

    @comp_timer.time_operation("krao")
    def krao_op(l, r): return khatri_rao_dense(l, r)

    @comp_timer.time_operation("add")
    def add_op(l, r): return l + r,

    @comp_timer.time_operation("matmul_sparse")
    def matmul_sparse_op(l, r): return ensure_csr(l) @ ensure_csc(r)

    @comp_timer.time_operation("hdmr_sparse")
    def hdmr_sparse_op(l, r): return ensure_csr(l).multiply(ensure_csr(r))

    @comp_timer.time_operation("kron_sparse")
    def kron_sparse_op(l, r): return sp.kron(ensure_csr(l), ensure_csc(r))

    @comp_timer.time_operation("krao_sparse")
    def krao_sparse_op(l, r): return sp.khatri_rao(ensure_csr(l), ensure_csc(r))

    ops = {
        "@": lambda l, r: matmul_op(l, r),
        "hdmr": lambda l, r: hdmr_op(l, r),
        "kron": lambda l, r: kron_op(l, r),
        "krao": lambda l, r: krao_op(l, r),
        "+": lambda l, r: add_op(l, r),

        "matmul_sparse": lambda l, r: matmul_sparse_op(l, r),
        "hdmr_sparse": lambda l, r: hdmr_sparse_op(l, r),
        "kron_sparse": lambda l, r: kron_sparse_op(l, r),
        "krao_sparse": lambda l, r: krao_sparse_op(l, r),

        "mat_trans": lambda l, r: l.T,
        "mat_trans_sparse": lambda l, r: l.T,
    }

    if node.node in ops:
        return ops[node.node](left_val, right_val)

    raise ValueError(f"Operation {node.node} not implemented in evaluator.")

def khatri_rao_dense(a, b):
    ma, na = a.shape
    mb, nb = b.shape
    assert na == nb, "Khatri-Rao requires matrices to have same number of columns"
    return (a.reshape(ma, 1, na) * b.reshape(1, mb, nb)).reshape(ma * mb, na)

def unfold_op_dist_over_add(graph: ExpressionTree) -> ExpressionTree:
    if graph.left is None and graph.right is None:
        return graph

    if isinstance(graph.left, ExpressionTree):
        graph.left = unfold_op_dist_over_add(graph.left)
    if isinstance(graph.right, ExpressionTree):
        graph.right = unfold_op_dist_over_add(graph.right)

    if (graph.node == BinaryMatrixOp.MATMUL.value \
        or graph.node == BinaryMatrixOp.KRON.value \
        or graph.node == BinaryMatrixOp.HDMR.value) \
        and isinstance(graph.right, ExpressionTree) \
        and graph.right.node == BinaryMatrixOp.MAT_ADD.value:

        return ExpressionTree(
            node=BinaryMatrixOp.MAT_ADD.value,
            left=ExpressionTree(
                node=graph.node,
                left=graph.left,
                right=graph.right.left,
            ),
            right=ExpressionTree(
                node=graph.node,
                left=graph.left,
                right=graph.right.right,
            )
        )

    if (graph.node == BinaryMatrixOp.MATMUL.value \
        or graph.node == BinaryMatrixOp.KRON.value \
        or graph.node == BinaryMatrixOp.HDMR.value) \
        and isinstance(graph.left, ExpressionTree) \
        and graph.left.node == BinaryMatrixOp.MAT_ADD.value:

        return ExpressionTree(
            node=BinaryMatrixOp.MAT_ADD.value,
            left=ExpressionTree(
                node=graph.node,
                left=graph.left.left,
                right=graph.right,
            ),
            right=ExpressionTree(
                node=graph.node,
                left=graph.left.right,
                right=graph.right,
            )
        )

    return graph

def lift_unary_ops(graph: ExpressionTree) -> ExpressionTree:
    if graph.left is None and graph.right is None:
        return unfold_unary_ops(graph.node)

    if isinstance(graph.left, ExpressionTree):
        graph.left = lift_unary_ops(graph.left)
    elif graph.left is not None:
        pass

    if isinstance(graph.right, ExpressionTree):
        graph.right = lift_unary_ops(graph.right)
    elif graph.right is not None:
        pass

    return graph

def unfold_unary_ops(leaf: str) -> ExpressionTree:
    unary_op_pattern = rf"\.({UnaryMatrixOp.ANY()}\(\))"
    unary_matches = list(re.finditer(unary_op_pattern, leaf))

    if not unary_matches:
        return ExpressionTree(node=leaf, left=None, right=None)

    targeted_matrix = re.match(rf"^({MATRIX_RAW})", leaf)
    if not targeted_matrix:
        return ExpressionTree(node=leaf, left=None, right=None)
    targeted_matrix = targeted_matrix.group(0)

    unary_ops = [match.group(1) for match in unary_matches]

    graph = ExpressionTree(node=targeted_matrix, left=None, right=None)
    for op in reversed(unary_ops):
        graph = ExpressionTree(node=op, left=graph, right=None)

    return graph