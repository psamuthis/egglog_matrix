from dataclasses import dataclass
from typing import Optional
from enum import Enum
import re


class BinaryMatrixOp(Enum):
    MATMUL = "@"
    MATMUL_SPARSE = "matmul_sparse"
    MAT_ADD = "+"
    MATADD_SPARSE = "matadd_sparse"
    KRON = "kron"
    KRON_SPARSE = "kron_sparse"
    KRAO = "krao"
    KRAO_SPARSE = "krao_sparse"
    HDMR = "hdmr"
    HDMR_SPARSE = "hdmr_sparse"
    MAT_CONCAT = "mat_concat"
    MAT_VEC_MUL = "mat_vec_mul"
    SPMV = "spmv"

    @classmethod
    def ANY(cls) -> str:
        op_patterns = [re.escape(op.value) for op in cls]
        return '|'.join(op_patterns)

class UnaryMatrixOp(Enum):
    MAT_INV = "mat_inv"
    MAT_MPINV = "mat_mpinv"
    MAT_TRANS = "mat_trans"
    TO_CSC = "to_CSC"
    TO_CSR = "to_CSR"

    @classmethod
    def ANY(cls) -> str:
        op_patterns = [re.escape(op.value) for op in cls]
        return '|'.join(op_patterns)

#MATRIX_TOKEN = r"Matrix\(\d+,\s*\d+,\s*\d*\.\d+\)(\.{UnaryMatrixOp.ANY()}\(\))?"
MATRIX_TOKEN = rf"Matrix\(\d+,\s*\d+,\s*\d*\.\d+\)(?:\.{UnaryMatrixOp.ANY()}\(\))*"

class DistributiveMatrixOp(Enum):
    MUL_LDIST_OVER_ADD = rf"""
        ({MATRIX_TOKEN})\s*
        {re.escape(BinaryMatrixOp.MATMUL.value)}\s*
        \(\s*
        ({MATRIX_TOKEN})\s*
        {re.escape(BinaryMatrixOp.MAT_ADD.value)}\s*
        ({MATRIX_TOKEN})\s*
        \)
    """

    MUL_RDIST_OVER_ADD = rf"(\(({MATRIX_TOKEN})\s*({BinaryMatrixOp.MAT_ADD})\s*({MATRIX_TOKEN})\)\s*({BinaryMatrixOp.MATMUL})\s*({MATRIX_TOKEN}))"

@dataclass
class ExpressionTree:
    node: str
    left: ExpressionTree | str | None
    right: ExpressionTree | str | None

    def print_tree(self, depth: int=0):
        indent = " " * depth * 4

        if self.left is None and self.right is None:
            print(f'{indent}C: {self.node}')
        else:
            print(f'{indent}OP: {self.node}')

        if isinstance(self.left, ExpressionTree):
            self.left.print_tree(depth + 1)
        elif self.left is not None:
            print(f'{indent}    LEFT (string): {self.left}')

        if isinstance(self.right, ExpressionTree):
            self.right.print_tree(depth + 1)
        elif self.right is not None:
            print(f'{indent}    RIGHT (string): {self.right}')

def build_compute_graph(expr: str, debug: str="", depth: int=0) -> ExpressionTree:
    indent = "-" * depth * 3

    balance = get_parenthesis_balance(expr)
    assert balance == 0, "build_compute_graph: input expr not balanced."

    pattern = rf"{BinaryMatrixOp.ANY()}"
    for match in re.finditer(pattern, expr):
        start,end = match.span()
        balance = get_parenthesis_balance_at(expr, start)

        if balance == 0:
            level_operation = match.group()
            left = trim_outer_parenthesis(expr[:start-1])
            right = trim_outer_parenthesis(expr[end:])

            left_tree = build_compute_graph(left, "left", depth+1)
            right_tree = build_compute_graph(right, "right", depth+1)

            return ExpressionTree(node=level_operation, left=left_tree, right=right_tree)

    return ExpressionTree(node=expr, left=None, right=None)

def unfold_expr(expr: str) -> str:
    expr = re.sub(DistributiveMatrixOp.MUL_LDIST_OVER_ADD.value, rf'(\1 @ \2) + (\1 @ \3)', expr, flags=re.VERBOSE)
    return expr

def get_parenthesis_balance(expr: str) -> int:
    balance = 0

    for i in range(0, len(expr)):
        if expr[i] == '(':
            balance = balance + 1
        elif expr[i] == ')':
            balance = balance - 1

    return balance

def trim_outer_parenthesis(expr: str) -> str:
    expr = expr.strip()

    while not bool(re.search(MATRIX_TOKEN+"$", expr)) and expr.startswith('(') and expr.endswith(')'):
        trimmed = expr[1:-1]

        if is_balanced(trimmed):
            expr = trimmed
        else:
            break

    return expr

def get_parenthesis_balance_at(expr: str, index: int) -> int:
    balance = 0

    for i in range(0, index):
        if expr[i] == '(':
            balance = balance + 1
        elif expr[i] == ')':
            balance = balance - 1

    return balance

def is_balanced(expr: str) -> bool:
    balance = get_parenthesis_balance(expr)
    return True if balance == 0 else False