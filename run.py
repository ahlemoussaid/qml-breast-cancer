"""
Breast cancer classification (Wisconsin diagnostic dataset):
logistic regression and RBF-SVM vs a quantum kernel SVM and a variational quantum classifier.
All models see the same 4 PCA features so the comparison is fair.
"""

import json
import os
import time

import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from quantum_models import VQC, QuantumKernel

SEED = 42
N_QUBITS = 4

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def load_data():
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target  # sklearn has 0 = malignant, flip it so malignant is the positive class
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    # standardise -> PCA down to one feature per qubit -> scale to [0, pi] for rotation angles
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(scaler.transform(X_tr))
    to_angle = MinMaxScaler((0, onp.pi)).fit(pca.transform(scaler.transform(X_tr)))

    def prep(X_):
        return onp.clip(to_angle.transform(pca.transform(scaler.transform(X_))), 0, onp.pi)

    print(f"train {len(y_tr)} / test {len(y_te)}, PCA keeps "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance")
    return prep(X_tr), prep(X_te), y_tr, y_te


def scores(y_true, y_pred, y_prob, seconds):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_malignant": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "seconds": round(seconds, 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot(results, loss_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(results)
    x = onp.arange(len(names))
    w = 0.27
    for i, (key, label) in enumerate([("accuracy", "Accuracy"),
                                      ("recall_malignant", "Recall (malignant)"),
                                      ("roc_auc", "ROC-AUC")]):
        ax1.bar(x + (i - 1) * w, [results[n][key] for n in names], w, label=label)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax1.set_ylim(0.7, 1.05)
    ax1.set_title(f"Test set ({N_QUBITS} PCA features / {N_QUBITS} qubits)")
    ax1.legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(range(1, len(loss_history) + 1), loss_history, marker="o", ms=3)
    ax2.set_title("VQC training loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Binary cross-entropy")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/comparison.png", dpi=150)
    plt.close()


def main():
    X_tr, X_te, y_tr, y_te = load_data()
    results = {}

    t = time.time()
    lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    results["Logistic Regression"] = scores(y_te, lr.predict(X_te), lr.predict_proba(X_te)[:, 1], time.time() - t)

    t = time.time()
    svm = SVC(kernel="rbf", probability=True, random_state=SEED).fit(X_tr, y_tr)
    results["SVM (RBF)"] = scores(y_te, svm.predict(X_te), svm.predict_proba(X_te)[:, 1], time.time() - t)

    t = time.time()
    kernel = QuantumKernel(N_QUBITS)
    qsvm = SVC(kernel="precomputed", probability=True, random_state=SEED).fit(kernel(X_tr, X_tr), y_tr)
    K_te = kernel(X_te, X_tr)
    results["Quantum Kernel SVM"] = scores(y_te, qsvm.predict(K_te), qsvm.predict_proba(K_te)[:, 1], time.time() - t)

    print("training VQC")
    t = time.time()
    vqc = VQC(N_QUBITS, n_layers=3, seed=SEED).fit(X_tr, y_tr)
    p = vqc.predict_proba(X_te)
    results["Variational Quantum Classifier"] = scores(y_te, (p > 0.5).astype(int), p, time.time() - t)

    print(f"\n{'model':<32}{'acc':>7}{'recall':>8}{'f1':>7}{'auc':>7}")
    for name, r in results.items():
        print(f"{name:<32}{r['accuracy']:7.3f}{r['recall_malignant']:8.3f}{r['f1']:7.3f}{r['roc_auc']:7.3f}")
    print(f"\nVQC trainable parameters: {vqc.n_params}")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    plot(results, vqc.loss_history)

    print("\nVQC circuit:")
    print(qml.draw(vqc.circuit, level="device")(vqc.weights, X_te[0]))


if __name__ == "__main__":
    main()
