"""Quantum kernel and variational classifier, both on N_QUBITS qubits."""

import numpy as onp
import pennylane as qml
from pennylane import numpy as pnp


class QuantumKernel:
    """Fidelity kernel k(a, b) = |<phi(a)|phi(b)>|^2 with a ZZ-type feature map."""

    def __init__(self, n_qubits, reps=2):
        self.n_qubits = n_qubits
        self.reps = reps
        dev = qml.device("default.qubit", wires=n_qubits)
        self._state = qml.QNode(self._state_circuit, dev)

    def feature_map(self, x):
        n = self.n_qubits
        for _ in range(self.reps):
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[i], wires=i)
            for i in range(n - 1):
                qml.IsingZZ(2 * (onp.pi - x[i]) * (onp.pi - x[i + 1]) / onp.pi, wires=[i, i + 1])

    def _state_circuit(self, x):
        self.feature_map(x)
        return qml.state()

    def __call__(self, A, B):
        # On hardware you'd run feature_map(a) + adjoint(feature_map)(b) and read P(|0..0>).
        # In simulation it's the same quantity, and much cheaper to get from the statevectors.
        SA = onp.array([self._state(a) for a in A])
        SB = onp.array([self._state(b) for b in B])
        return onp.abs(SA.conj() @ SB.T) ** 2


class VQC:
    """RY angle encoding + StronglyEntanglingLayers, prediction from <Z> on qubit 0."""

    def __init__(self, n_qubits, n_layers=3, seed=42):
        self.n_qubits = n_qubits
        rng = onp.random.default_rng(seed)
        self.weights = pnp.array(0.1 * rng.standard_normal((n_layers, n_qubits, 3)), requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)
        self.rng = rng
        self.loss_history = []
        dev = qml.device("default.qubit", wires=n_qubits)
        self.circuit = qml.QNode(self._circuit, dev, diff_method="backprop")

    def _circuit(self, weights, x):
        # x is (n_qubits,) or (n_qubits, batch) -> the whole batch goes through in one call
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    def _prob(self, weights, bias, X):
        z = self.circuit(weights, pnp.array(X, requires_grad=False).T)
        return (1 - pnp.tanh(z + bias)) / 2

    def _loss(self, weights, bias, X, y):
        p = pnp.clip(self._prob(weights, bias, X), 1e-6, 1 - 1e-6)
        return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))

    def fit(self, X, y, epochs=25, batch_size=32, lr=0.1, verbose=True):
        opt = qml.AdamOptimizer(stepsize=lr)
        w, b = self.weights, self.bias
        for epoch in range(epochs):
            idx = self.rng.permutation(len(y))
            for start in range(0, len(idx), batch_size):
                bi = idx[start:start + batch_size]
                (w, b, _, _), _ = opt.step_and_cost(self._loss, w, b, X[bi], y[bi])
            loss = float(self._loss(w, b, X, y))
            self.loss_history.append(loss)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch + 1:>2}/{epochs}  loss {loss:.4f}")
        self.weights, self.bias = w, b
        return self

    def predict_proba(self, X):
        return onp.array(self._prob(self.weights, self.bias, X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    @property
    def n_params(self):
        return self.weights.size + 1
