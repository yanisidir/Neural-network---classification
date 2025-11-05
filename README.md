# Neural Network – Classification

[![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-requirements.txt-green.svg)](./requirements.txt)

This project implements and studies neural networks for **image classification** on the **MNIST dataset**.
It is divided into two main parts:

1. **From scratch implementation** (using only NumPy and Pandas)
2. **Keras / TensorFlow implementation** (leveraging high-level libraries for faster experimentation)

---

## 📂 Repository structure

```
Neural network - classification/
│
├── from_scratch_nn.py        # Full NumPy-only implementation (forward, backward, training loop)
├── keras_nn.py               # Keras/TensorFlow implementation with different architectures
├── requirements.txt          # Python dependencies
├── figures/                  # Plots 
└── README.md                 # Project documentation
```

---

## 🧩 Part 1: Neural Network from Scratch

* Dataset: **MNIST handwritten digits (train.csv)**
* Features implemented manually:

  * Dataset loading and preprocessing (normalization, shuffling, train/dev split)
  * Forward propagation (ReLU + Softmax)
  * Backward propagation (gradients via cross-entropy loss)
  * Parameter updates with gradient descent
  * Training loop with accuracy reporting
  * Saving trained weights and biases to `.npz` file

**Architecture:**

* Input layer: **784 neurons** (28×28 pixels)
* Hidden layer: **10 neurons, ReLU activation**
* Output layer: **10 neurons, softmax activation**

---

## 🤖 Part 2: Neural Network with Keras/TensorFlow

Using **TensorFlow/Keras**, different architectures are built and trained:

* **Baseline network**: one hidden layer of 10 neurons
* **Variation in learning rate**: 0.01 vs 0.2
* **Wider hidden layer**: 500 neurons
* **Deeper network**: 2 hidden layers (700 and 500 neurons)
* **Mini-batch training**: smaller batch size to study overfitting

**Training details:**

* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Metric: **Accuracy**
* Up to **300 epochs**
* **Validation accuracy** and **loss curves** are saved as plots

---

## 📊 Results

* Training and validation curves (loss and accuracy) are generated for each experiment.
* Maximum validation accuracy is printed for each tested architecture and learning rate.
* Overfitting detection is performed when using small batch sizes.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yanisidir/Neural-network-classification.git
cd Neural-network-classification
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the from-scratch implementation

```bash
python from_scratch_nn.py
```

### Run the Keras/TensorFlow experiments

```bash
python keras_nn.py
```

All plots will be saved automatically in figures directory.

---

## 📦 Requirements

See [requirements.txt](./requirements.txt). Main dependencies:

* numpy
* pandas
* matplotlib
* tensorflow (with keras)

---

## 📌 Notes

* This project is educational: the first part shows how to implement a simple neural network step by step.
* The second part highlights how higher-level libraries simplify architecture exploration and allow focus on **hyperparameter tuning** and **model comparison**.

