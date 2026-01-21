from egglog import Expr
from Parser import build_compute_graph, unfold_expr

def compile(expr, egraph):
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    optimized_expr = str(optimized_expr)
    print(f'Input expr: {optimized_expr}')

    optimized_expr = unfold_expr(optimized_expr)
    compute_graph = build_compute_graph(optimized_expr)
    compute_graph.print_tree()