#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------
def ensure_dir(d="figures"):
    os.makedirs(d, exist_ok=True)
    return d

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def plot_history(history, title_prefix, outdir):
    hist = history.history
    # Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist['loss'], label='train loss')
    plt.plot(hist['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} — Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist['accuracy'], label='train acc')
    plt.plot(hist['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{title_prefix} — Accuracy')
    plt.legend()
    savefig(os.path.join(outdir, f"{title_prefix.replace(' ', '_')}.png"))

def summarize_best(history, label=""):
    hist = history.history
    val_acc = np.array(hist['val_accuracy'])
    best_idx = int(val_acc.argmax())
    best = float(val_acc[best_idx])
    print(f"[{label}] best val_accuracy = {best*100:.2f}% at epoch {best_idx}")
    return best, best_idx

def detect_overfitting_epoch(val_losses, patience=5, tol=0.0):
    """
    Heuristique : première époque e où la perte de validation est
    supérieure au min courant + tol, pendant 'patience' époques consécutives.
    Renvoie l'époque e (index int) ou None si non détecté.
    """
    best = np.inf
    wait = 0
    start = None
    for i, v in enumerate(val_losses):
        if v < best - tol:
            best = v
            wait = 0
            start = None
        else:
            wait += 1
            if start is None:
                start = i
            if wait >= patience:
                return start - (patience - 1)
    return None

# ---------------------------------------------------------------------
# 1) Données MNIST
# ---------------------------------------------------------------------
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalisation
X_train = (X_train / 255.0).astype('float32')
X_test  = (X_test  / 255.0).astype('float32')

# Aplatissement (28,28) -> (784,)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)

# One-hot
Y_train_oh = to_categorical(Y_train, 10)
Y_test_oh  = to_categorical(Y_test, 10)

print("Train shapes:", X_train.shape, Y_train_oh.shape)
print("Test  shapes:", X_test.shape,  Y_test_oh.shape)

outdir = ensure_dir()

# ---------------------------------------------------------------------
# 2) Modèle de base : 784 → 10 (ReLU) → 10 (softmax), Adam lr=1e-2
# ---------------------------------------------------------------------
model = Sequential([
    Input(shape=(784,)),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
])

opt = Adam(learning_rate=1e-2)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement 300 epochs, batch_size = len(X_train) (full-batch)
out_base = model.fit(
    X_train, Y_train_oh,
    batch_size=X_train.shape[0],
    epochs=300,
    validation_data=(X_test, Y_test_oh),
    verbose=0
)

print("Clés out.history (modèle base):", list(out_base.history.keys()))
plot_history(out_base, "Base_lr1e-2_fullbatch_300epochs", outdir)
best_base, best_idx_base = summarize_best(out_base, "base lr=1e-2")

# Vérification convergence: variation dernière vs avant-dernière val_acc
va = out_base.history['val_accuracy']
if len(va) >= 2:
    print(f"[base] Δval_acc(last-1,last) = {va[-1]-va[-2]:.4e}")

# (Optionnel) allonger jusqu’à convergence ~1e-3 près :
# -> Ici on illustre juste la condition ; si non convergé, relancer .fit(...) quelques epochs.

# ---------------------------------------------------------------------
# 3) Même modèle, lr = 0.2
# ---------------------------------------------------------------------
model_lr = Sequential([
    Input(shape=(784,)),
    Dense(10, activation='relu'),
    Dense(10, activation='softmax')
])
opt = Adam(learning_rate=0.2)
model_lr.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

out_lr = model_lr.fit(
    X_train, Y_train_oh,
    batch_size=X_train.shape[0],
    epochs=300,
    validation_data=(X_test, Y_test_oh),
    verbose=0
)
plot_history(out_lr, "Base_lr0.2_fullbatch_300epochs", outdir)
best_lr, best_idx_lr = summarize_best(out_lr, "base lr=0.2")

# ---------------------------------------------------------------------
# 4) Couche cachée beaucoup plus large (×50 ~ 500 neurones), lr=1e-2
# ---------------------------------------------------------------------
model_wide = Sequential([
    Input(shape=(784,)),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])
opt = Adam(learning_rate=1e-2)
model_wide.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

out_wide = model_wide.fit(
    X_train, Y_train_oh,
    batch_size=X_train.shape[0],
    epochs=300,
    validation_data=(X_test, Y_test_oh),
    verbose=0
)
plot_history(out_wide, "Wide500_lr1e-2_fullbatch_300epochs", outdir)
best_wide, best_idx_wide = summarize_best(out_wide, "wide (500) lr=1e-2")

# ---------------------------------------------------------------------
# 5) Réseau plus profond (700 → 500) avec ReLU, 200 epochs, lr=1e-2
# ---------------------------------------------------------------------
model_deep = Sequential([
    Input(shape=(784,)),
    Dense(700, activation='relu'),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])
opt = Adam(learning_rate=1e-2)
model_deep.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

out_deep = model_deep.fit(
    X_train, Y_train_oh,
    batch_size=X_train.shape[0],
    epochs=200,
    validation_data=(X_test, Y_test_oh),
    verbose=0
)
plot_history(out_deep, "Deep700_500_lr1e-2_fullbatch_200epochs", outdir)
best_deep, best_idx_deep = summarize_best(out_deep, "deep (700,500) lr=1e-2")

# ---------------------------------------------------------------------
# 6) Même réseau profond, batch_size ÷ 10 (mini-batch), 200 epochs
#    → étude sur-apprentissage via courbe val_loss
# ---------------------------------------------------------------------
out_deep_mb = model_deep.fit(
    X_train, Y_train_oh,
    batch_size=max(1, X_train.shape[0] // 10),
    epochs=200,
    validation_data=(X_test, Y_test_oh),
    verbose=0
)
# Trace seulement la val_loss comme demandé
plt.figure(figsize=(8, 5))
plt.plot(out_deep_mb.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs (deep, mini-batch)')
plt.legend()
savefig(os.path.join(outdir, "Deep_minibatch_val_loss.png"))

# Détection de l’époque d’entrée en sur-apprentissage (heuristique)
overfit_epoch = detect_overfitting_epoch(
    np.array(out_deep_mb.history['val_loss']),
    patience=5, tol=0.0
)
if overfit_epoch is None:
    print("Sur-apprentissage non détecté de façon claire (selon l’heuristique choisie).")
else:
    print(f"Sur-apprentissage détecté (heuristique) à partir de l’époque ~ {overfit_epoch}")

# ---------------------------------------------------------------------
# 7) Résumé des meilleurs scores
# ---------------------------------------------------------------------
print("\n=== RÉSUMÉ DES MEILLEURES VAL_ACC ===")
print(f"Base lr=1e-2    : {best_base*100:.2f}% @ epoch {best_idx_base}")
print(f"Base lr=0.2     : {best_lr*100:.2f}% @ epoch {best_idx_lr}")
print(f"Wide (500) lr=1e-2 : {best_wide*100:.2f}% @ epoch {best_idx_wide}")
print(f"Deep (700,500) lr=1e-2 : {best_deep*100:.2f}% @ epoch {best_idx_deep}")
print("Figures enregistrées dans:", outdir)
