from egglog import Expr
from Parser import build_compute_graph, BinaryMatrixOp, UnaryMatrixOp, ExpressionTree

def compile(expr, egraph):
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    optimized_expr = str(optimized_expr)
    print(f'Input expr: {optimized_expr}')

    compute_graph = build_compute_graph(optimized_expr)
    compute_graph.print_tree()

    compute_graph = unfold_op_dist_over_add(compute_graph)
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