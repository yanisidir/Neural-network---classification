#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from pathlib import Path

out = Path("figures")
out.mkdir(exist_ok=True)

# =========================
# 1) Chargement & split
# =========================
file_path = 'train.csv'  # MNIST Kaggle-like (label + 784 pixels)
df = pd.read_csv(file_path)
data = df.to_numpy()

print("Dimensions (n_images, 785):", data.shape)
print("→ 785 = 1 label + 784 pixels (28x28)")

# Shuffle global
rng = np.random.default_rng(42)
rng.shuffle(data, axis=0)

# Split
X_dev  = data[:1000, 1:].astype(np.float32)
Y_dev  = data[:1000, 0].astype(np.int64)
X_train = data[1000:, 1:].astype(np.float32)
Y_train = data[1000:, 0].astype(np.int64)

# Normalisation 0..1
X_dev  /= 255.0
X_train /= 255.0

print("Train:", X_train.shape, Y_train.shape)
print("Dev  :", X_dev.shape, Y_dev.shape)

# =========================
# 2) Réseau (784 → 10 → 10)
# =========================
INPUT_SIZE  = 784
HIDDEN_SIZE = 10
OUTPUT_SIZE = 10

def initialize_parameters(input_size, hidden_size, output_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    W0 = rng.uniform(-0.5, 0.5, size=(hidden_size, input_size)).astype(np.float32)
    b0 = rng.uniform(-0.5, 0.5, size=(hidden_size, 1)).astype(np.float32)
    W1 = rng.uniform(-0.5, 0.5, size=(output_size, hidden_size)).astype(np.float32)
    b1 = rng.uniform(-0.5, 0.5, size=(output_size, 1)).astype(np.float32)
    return W0, b0, W1, b1

def relu(Z):
    return np.maximum(Z, 0.0)

def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)

def softmax(Z):
    # Z: (C, N). Numériquement stable
    Z_shift = Z - Z.max(axis=0, keepdims=True)
    expZ = np.exp(Z_shift, dtype=np.float64)
    return (expZ / expZ.sum(axis=0, keepdims=True)).astype(np.float32)

def one_hot(y, num_classes=10):
    # y: (N,)
    Y = np.zeros((num_classes, y.size), dtype=np.float32)
    Y[y, np.arange(y.size)] = 1.0
    return Y

def forward_propagation(X, W0, b0, W1, b1):
    """
    X: (N, 784)  → on travaille en colonnes : X^T = (784, N)
    Retourne dictionnaire d’activation/linéaire pour backprop.
    """
    X_col = X.T  # (784, N)
    Z0 = W0 @ X_col + b0           # (10, N)
    A0 = relu(Z0)                  # (10, N)
    Z1 = W1 @ A0 + b1              # (10, N)
    A1 = softmax(Z1)               # (10, N)
    cache = {"X": X_col, "Z0": Z0, "A0": A0, "Z1": Z1, "A1": A1}
    return A1, cache

def backward_propagation(Y, cache, W0, W1):
    """
    Y: (10, N) one-hot
    cache: dict de forward
    Retourne dJ/dW0, dJ/db0, dJ/dW1, dJ/db1 (full-batch, moyenne sur N)
    """
    X  = cache["X"]                 # (784, N)
    Z0 = cache["Z0"]                # (10, N)
    A0 = cache["A0"]                # (10, N)
    A1 = cache["A1"]                # (10, N)
    N  = X.shape[1]

    # Cross-entropy + softmax → gradient simple: A1 - Y
    dZ1 = A1 - Y                    # (10, N)
    dW1 = (dZ1 @ A0.T) / N          # (10, 10)
    db1 = dZ1.mean(axis=1, keepdims=True)  # (10,1)

    dA0 = W1.T @ dZ1                # (10, N)
    dZ0 = dA0 * relu_derivative(Z0) # (10, N)
    dW0 = (dZ0 @ X.T) / N           # (10, 784)
    db0 = dZ0.mean(axis=1, keepdims=True)  # (10,1)

    return dW0.astype(np.float32), db0.astype(np.float32), dW1.astype(np.float32), db1.astype(np.float32)

def update_parameters(W0, b0, W1, b1, dW0, db0, dW1, db1, lr):
    W0 -= lr * dW0
    b0 -= lr * db0
    W1 -= lr * dW1
    b1 -= lr * db1
    return W0, b0, W1, b1

def predict_classes(X, W0, b0, W1, b1):
    A1, _ = forward_propagation(X, W0, b0, W1, b1)  # A1: (10, N)
    return np.argmax(A1, axis=0)  # (N,)

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

# =========================
# 3) Entraînement full-batch
# =========================
def train(X_train, Y_train, X_dev, Y_dev, iters=100, lr=1.0, seed=0):
    rng = np.random.default_rng(seed)
    W0, b0, W1, b1 = initialize_parameters(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, rng=rng)

    Y_train_oh = one_hot(Y_train, OUTPUT_SIZE)  # (10, Ntrain)

    for it in range(1, iters + 1):
        # Forward
        A1, cache = forward_propagation(X_train, W0, b0, W1, b1)

        # Loss (optionnel) : cross-entropy moyenne
        # eps pour stabilité
        eps = 1e-12
        ce = -np.sum(Y_train_oh * np.log(A1 + eps)) / X_train.shape[0]

        # Backward
        dW0, db0, dW1, db1 = backward_propagation(Y_train_oh, cache, W0, W1)

        # Update
        W0, b0, W1, b1 = update_parameters(W0, b0, W1, b1, dW0, db0, dW1, db1, lr)

        if it % 10 == 0 or it == 1 or it == iters:
            train_acc = accuracy(predict_classes(X_train, W0, b0, W1, b1), Y_train)
            dev_acc   = accuracy(predict_classes(X_dev,   W0, b0, W1, b1), Y_dev)
            print(f"[{it:3d}/{iters}] loss={ce:.4f} | acc_train={train_acc*100:.2f}% | acc_dev={dev_acc*100:.2f}%")

    # Sauvegarde des paramètres
    np.savez('trained_parameters.npz', W0=W0, b0=b0, W1=W1, b1=b1)
    return W0, b0, W1, b1

# Lancement
W0, b0, W1, b1 = train(X_train, Y_train, X_dev, Y_dev, iters=100, lr=1.0, seed=0)

# (Bonus) Fine-tuning 100 itérations à lr=0.1 — décommente si souhaité
# W0, b0, W1, b1 = train(X_train, Y_train, X_dev, Y_dev, iters=100, lr=0.1, seed=0)
