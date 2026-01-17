from egglog import Expr
from Parser import build_compute_graph

def compile(expr, egraph):
    egraph.saturate(visualize=False)
    optimized_expr = egraph.extract(expr)
    print(f'Input expr: {optimized_expr}')

    compute_graph = build_compute_graph(str(optimized_expr))
    compute_graph.print_tree()