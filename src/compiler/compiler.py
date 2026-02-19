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

from .parser import build_compute_graph
from .parser import BinaryMatrixOp
from .parser import UnaryMatrixOp
from .parser import ExpressionTree
from .parser import MATRIX_TOKEN
from .parser import MATRIX_RAW

def compile(expr, egraph, matrices_data=None, DEBUG=False):
    if(DEBUG):
        print(f"Rewriting {egraph.extract(expr)} with egglog...")
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    optimized_expr = str(optimized_expr)
    if(DEBUG):
        print(f'Rewrote expr to: {optimized_expr}')

    if(DEBUG):
        print(f'\nBuilding compute graph...')
    compute_graph = build_compute_graph(optimized_expr)
    compute_graph = unfold_op_dist_over_add(compute_graph)
    compute_graph = lift_unary_ops(compute_graph)
    if(DEBUG):
        print(f'Resulting compute graph:')
        compute_graph.print_tree()

    return evaluate(compute_graph, matrices_data)

import numpy as np
import scipy.sparse as sp

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

    def ensure_sparse(val):
        return val if sp.issparse(val) else sp.csr_matrix(val)

    def ensure_dense(val):
        return val.toarray() if sp.issparse(val) else val

    ops = {
        "@": lambda l, r: l @ r,
        "matmul": lambda l, r: l @ r,
        "hdmr": lambda l, r: l * r,
        "kron": lambda l, r: np.kron(l, r),
        "krao": lambda l, r: khatri_rao_dense(l, r),
        "+": lambda l, r: l + r,

        "matmul_sparse": lambda l, r: ensure_sparse(l) @ ensure_sparse(r),
        "hdmr_sparse": lambda l, r: ensure_sparse(l).multiply(ensure_sparse(r)),
        "kron_sparse": lambda l, r: sp.kron(ensure_sparse(l), ensure_sparse(r)),
        "krao_sparse": lambda l, r: sp.khatri_rao(ensure_sparse(l), ensure_sparse(r)),

        "mat_trans()": lambda l, r: l.T,
        "mat_trans_sparse()": lambda l, r: l.T,
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
        and isinstance(graph.right, ExpressionTree):

        if graph.right.node == BinaryMatrixOp.MAT_ADD.value:
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
        and isinstance(graph.left, ExpressionTree):

        if graph.left.node == BinaryMatrixOp.MAT_ADD.value:
            return ExpressionTree(
                node=BinaryMatrixOp.MAT_ADD.value,
                left=ExpressionTree(
                    node=BinaryMatrixOp.MATMUL.value,
                    left=graph.left.left,
                    right=graph.right,
                ),
                right=ExpressionTree(
                    node=BinaryMatrixOp.MATMUL.value,
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