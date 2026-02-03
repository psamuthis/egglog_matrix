from egglog import Expr
import re

from Parser import build_compute_graph
from Parser import BinaryMatrixOp
from Parser import UnaryMatrixOp
from Parser import ExpressionTree
from Parser import MATRIX_TOKEN

def compile(expr, egraph):
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    optimized_expr = str(optimized_expr)
    print(f'Input expr: {optimized_expr}')

    print(f'After parsing.')
    compute_graph = build_compute_graph(optimized_expr)
    compute_graph.print_tree()

    print(f'After operations unfolding.')
    compute_graph = unfold_op_dist_over_add(compute_graph)
    compute_graph.print_tree()

    print(f'After unary lifting.')
    compute_graph = lift_unary_ops(compute_graph)
    compute_graph.print_tree()

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

    targeted_matrix = re.match(rf"^({MATRIX_TOKEN.split(").")[0]})", leaf)
    if not targeted_matrix:
        return ExpressionTree(node=leaf, left=None, right=None)
    targeted_matrix = targeted_matrix.group(1)+")"

    unary_ops = [match.group(1) for match in unary_matches]

    graph = ExpressionTree(node=targeted_matrix, left=None, right=None)
    for op in reversed(unary_ops):
        graph = ExpressionTree(node=op, left=graph, right=None)

    return graph