# Deriving Backpropagation with Cross-Entropy Loss  
### Minimizing the Loss for Classification Models  
 
**Based on:** *Implementing Backpropagation From Scratch in Python 3+*   

---

## 🧠 Overview

This project demonstrates the **derivation and implementation of backpropagation** in a **feedforward neural network**, using **Cross-Entropy Loss** as the objective function. The derivation covers both **categorical** and **multi-class cross-entropy** cases and connects the theory directly to a working Python implementation.

By the end, you will understand how the gradients of the weights and biases are computed and used in **gradient descent** to minimize the loss in classification models.

---

## 📘 Contents

1. [Introduction](#introduction)  
2. [Cross-Entropy Loss Derivation](#cross-entropy-loss-derivation)  
   - [Categorical Cross-Entropy (Softmax)](#categorical-cross-entropy-softmax)  
   - [Multi-Class Cross-Entropy (Sigmoid)](#multi-class-cross-entropy-sigmoid)  
3. [Backpropagation Derivation](#backpropagation-derivation)  
4. [Python Implementation](#python-implementation)  
   - [Network Initialization](#network-initialization)  
   - [Forward Pass](#forward-pass)  
   - [Backward Pass](#backward-pass)  
   - [Gradient Descent](#gradient-descent)  
5. [Training Example](#training-example)  
6. [References](#references)

---

## 🧩 Introduction

There are many possible **loss functions** for neural networks, but when it comes to **classification**, **Cross-Entropy Loss** is the most commonly used due to its probabilistic interpretation and compatibility with Softmax and Sigmoid activations.

The goal of backpropagation is to **adjust the weights (W)** and **biases (B)** in the network to minimize this loss.

---

## 📊 Cross-Entropy Loss Derivation

### Categorical Cross-Entropy (Softmax)
Used when each input belongs to exactly one class.  
The true labels \( y \) are **one-hot encoded**, and the last layer uses a **Softmax** activation to convert logits into probabilities.

\[
J = - \sum_{m} y_m \log(a_m)
\]

where:
- \( a_m = \frac{e^{z_m}}{\sum_k e^{z_k}} \)
- \( z_m \) is the pre-activation for neuron \( m \)
- \( a_m \) is the Softmax output (probability)

Through the chain rule, the gradient for the last layer simplifies to:

\[
\delta^H = a^H - y
\]

This result elegantly connects the output probabilities to the true labels.

---

### Multi-Class Cross-Entropy (Sigmoid)
When multiple classes can be active simultaneously (e.g., an image containing both a cat and a dog), we use **Sigmoid activations** instead of Softmax.

\[
J = - \sum_m \left[ y_m \log(a_m) + (1 - y_m)\log(1 - a_m) \right]
\]
\[
a_m = \sigma(z_m)
\]

By differentiating and applying the chain rule, we obtain the same result as with Softmax:

\[
\delta^H = a^H - y
\]

Thus, the **backpropagation equations** remain consistent across both categorical and multi-class cases.

---

## ⚙️ Backpropagation Derivation

The backpropagation algorithm consists of two passes:

1. **Forward Pass:** Compute activations \( a^L \) and pre-activations \( z^L \) for each layer.
2. **Backward Pass:** Propagate the error \( \delta^L \) backward to compute gradients:
   - \( \delta^L = (W^{L+1})^T \delta^{L+1} * \sigma'(z^L) \)
   - \( \frac{\partial J}{\partial W^L} = \delta^L (a^{L-1})^T \)
   - \( \frac{\partial J}{\partial b^L} = \delta^L \)

---

## 🧮 Python Implementation

### Network Initialization
We define a `Network` class with the following structure:
```python
model = Network([784, 30, 10])
```
# This creates:

Input layer: 784 neurons

Hidden layer: 30 neurons

Output layer: 10 neurons

Weights (Wₙ) and biases (Bₙ) are initialized randomly from a standard normal distribution using NumPy’s randn().

# Forward Pass

Compute:

**Sigmoid** is used as the activation function:

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Backward Pass

The backward pass calculates gradients layer-by-layer using stored activations and pre-activations.

delta = (a_H - y) * sigmoid_derivative(z_H)


The gradients are stored as:

dW[L] = delta @ a_prev.T
db[L] = delta

Gradient Descent

After accumulating gradients across a mini-batch, we update parameters:

where 
𝜂
η is the learning rate.

# 🧠 Training Example

A training loop might look like:
```bash
for epoch in range(epochs):
    for mini_batch in dataset:
        network.gradient_descent(mini_batch, learning_rate)
```

Over multiple epochs, the weights and biases converge to values that minimize the classification loss.

# 📚 References

- Essam Wisam, Deriving Backpropagation with Cross-Entropy Loss
[Medium Article, October 2, 2021]

- Cute Dogs & Cats — Illustrations referenced in the original post.

# 💡 Key Takeaways

- Cross-Entropy Loss aligns the model’s probability distribution with the true distribution.

- Backpropagation with Softmax or Sigmoid yields the same simplified gradient at the output layer.

Implementing the algorithm from scratch helps demystify how neural networks actually learn.

# 🧩 Author
- Simanga Mchunu
*“Derive, don’t just use — understand the math behind the magic.”*