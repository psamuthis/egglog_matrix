import unittest
from Rewrites import egraph, Matrix, Vector, SPARSITY_THRESHOLD, StorageFormat

class TestMatrixMatrix(unittest.TestCase):
    def test_matmul(self):
        egraph.push()
        input = egraph.let(
            "associativity-example",
            (Matrix(64, 8, 0.0, StorageFormat.NATIVE.value) @
            Matrix(8, 256, 0.0, StorageFormat.NATIVE.value)) @
            Matrix(256, 2, 0.0, StorageFormat.NATIVE.value)
        )
        expected = egraph.let(
            "expected",
            Matrix(64, 8, 0.0, StorageFormat.NATIVE.value) @
            (Matrix(8, 256, 0.0, StorageFormat.NATIVE.value) @
            Matrix(256, 2, 0.0, StorageFormat.NATIVE.value))
        )

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(input == expected))
        egraph.pop()

    def test_kronecker_left_distributivity(self):
        """Test: x.kron(y.matadd(z)) -> (x.kron(y)).matadd(x.kron(z))"""
        egraph.push()

        x = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", x.kron(y.mat_add(z)))
        right_side = egraph.let("right", (x.kron(y)).mat_add(x.kron(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_right_distributivity(self):
        """Test: (y.matadd(z)).kron(x) -> (y.kron(x)).matadd(z.kron(x))"""
        egraph.push()

        x = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", y.mat_add(z).kron(x))
        right_side = egraph.let("right", (y.kron(x)).mat_add(z.kron(x)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_associativity(self):
        """Test: (x.kron(y)).kron(z) -> x.kron((y.kron(z)))"""
        egraph.push()

        x = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(6, 7, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", (x.kron(y)).kron(z))
        right_side = egraph.let("right", x.kron(y.kron(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_mixed_product(self):
        """Test: (w.kron(x)) @ (y.kron(z)) -> (w@y).kron(x@z)"""
        egraph.push()

        w = Matrix(2, 3, 0.0, StorageFormat.NATIVE.value)
        x = Matrix(4, 5, 0.0, StorageFormat.NATIVE.value)
        y = Matrix(3, 2, 0.0, StorageFormat.NATIVE.value)
        z = Matrix(5, 4, 0.0, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", (w.kron(x)) @ (y.kron(z)))
        right_side = egraph.let("right", (w @ y).kron(x @ z))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_hadamard_distributivity(self):
        """Test: (w.kron(x)).hdmr((y.kron(z))) -> (w.hdmr(y)).kron((x.hdmr(z)))"""
        egraph.push()

        w = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_inverse(self):
        """Test: (w.kron(x)).mat_inv() -> w.mat_inv().kron(x.mat_inv())"""
        egraph.push()

        w = Matrix(3, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(2, 2, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.kron(x).mat_inv())
        right_side = egraph.let("right", w.mat_inv().kron(x.mat_inv()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_transpose_distributivity(self):
        egraph.push()
        w = Matrix(3, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(2, 2, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.kron(x).mat_trans())
        right_side = egraph.let("right", w.mat_trans().kron(x.mat_trans()))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_krao_transpose_times_krao(self):
        egraph.push()
        w = Matrix(3, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(2, 2, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.krao(x).mat_trans() @ w.krao(x))
        right_side = egraph.let("right", (w.mat_trans()@w).hdmr(x.mat_trans()@x))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kron_times_krao_distributivity(self):
        egraph.push()
        w = Matrix(3, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(2, 2, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(4, 2, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(3, 2, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", (w.kron(x)) @ (y.krao(z)))
        right_side = egraph.let("right", (w@y).krao(x@z))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hadamard_commutativity(self):
        """Test: w.hdmr(x) -> x.hdmr(w)"""
        egraph.push()

        w = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.hdmr(x))
        right_side = egraph.let("right", x.hdmr(w))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hadamard_associativity(self):
        """Test: w.hdmr(x.hdmr(y)) -> (w.hdmr(x)).hdmr(y)"""
        egraph.push()

        w = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.hdmr(x.hdmr(y)))
        right_side = egraph.let("right", (w.hdmr(x)).hdmr(y))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hadamard_distributivity_over_addition(self):
        """Test: w.hdmr(x.mat_add(y)) -> (w.hdmr(x)).mat_add(w.hdmr(y))"""
        egraph.push()

        w = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(3, 4, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", w.hdmr(x.mat_add(y)))
        right_side = egraph.let("right", (w.hdmr(x)).mat_add(w.hdmr(y)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_hdmr_kron_hdmr(self):
        egraph.push()
        w = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)  # same dims as w
        z = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)  # same dims as x

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertTrue(egraph.check_bool(left_side == right_side))
        egraph.pop()

    def test_kronecker_hadamard_distributivity_fails_wrong_dims(self):
        egraph.push()

        w = Matrix(2, 3, 0.5, StorageFormat.NATIVE.value)
        x = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)
        y = Matrix(3, 2, 0.5, StorageFormat.NATIVE.value)
        z = Matrix(4, 5, 0.5, StorageFormat.NATIVE.value)

        left_side = egraph.let("left", (w.kron(x)).hdmr(y.kron(z)))
        right_side = egraph.let("right", (w.hdmr(y)).kron(x.hdmr(z)))

        egraph.saturate(visualize=False)
        self.assertFalse(egraph.check_bool(left_side == right_side))
        egraph.pop()