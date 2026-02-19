from __future__ import annotations
from egglog import *

class Vector(Expr):
    def __init__(self, len: i64Like) -> None: ...

    @property
    def len(self) -> i64: ...