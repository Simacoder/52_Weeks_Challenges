# 🧠 Recreating PyTorch from Scratch (with GPU Support and Automatic Differentiation)

Build your own deep learning framework — **from scratch** — using **C/C++**, **CUDA**, and **Python**.  
This project, nicknamed **norch**, is a minimalist reimplementation of core PyTorch concepts, featuring:

- ✅ **GPU acceleration** via CUDA  
- 🔁 **Automatic differentiation**  
- 🧩 **Modular neural network components** (Layers, Activations, Losses)  
- ⚙️ **Optimizers** (including SGD with Momentum)  
- 🧪 **Training loop support** with a clean, PyTorch-like API

---

## 🚀 Overview

The goal of this project is to understand **how PyTorch works under the hood** by implementing its fundamental building blocks:
- Tensor operations on CPU and GPU
- Computation graph construction for automatic differentiation
- Parameterized layers (`Linear`, `Sigmoid`, etc.)
- Loss functions (e.g., `MSELoss`)
- Optimizers (e.g., `SGD`)
- Device management and tensor movement (`.to("cuda")`)

By the end, you’ll have a working framework that can **train neural networks** on your GPU just like PyTorch.

---

## 🧩 Project Structure
```bash
norch/
├── init.py
├── tensor.py # Core tensor class with autodiff support
├── nn/
│ ├── init.py
│ ├── module.py # Base Module class
│ ├── linear.py # Linear layer implementation
│ ├── activations.py # Activation functions (e.g., Sigmoid)
│ └── losses.py # Loss functions (e.g., MSELoss)
└── optim/
├── init.py
└── sgd.py # SGD optimizer with momentum

```
---

## ⚙️ Example: Training a Simple Model

Below is a minimal example of training a feedforward neural network using **norch**.
```bash
import norch
import norch.nn as nn
import norch.optim as optim
import random
import math

random.seed(1)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

device = "cuda"
epochs = 10

model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

x_values = [i * 0.4 for i in range(51)]
y_true = [math.pow(math.sin(x), 2) for x in x_values]

for epoch in range(epochs):
    for x, target in zip(x_values, y_true):
        x = norch.Tensor([[x]]).T.to(device)
        target = norch.Tensor([[target]]).T.to(device)

        outputs = model(x)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}")
```
## 🧮 Sample Output
```bash
Epoch [1/10], Loss: 1.7035
Epoch [2/10], Loss: 0.7193
Epoch [3/10], Loss: 0.3068
Epoch [4/10], Loss: 0.1742
Epoch [5/10], Loss: 0.1342
Epoch [6/10], Loss: 0.1232
Epoch [7/10], Loss: 0.1220
Epoch [8/10], Loss: 0.1241
Epoch [9/10], Loss: 0.1270
Epoch [10/10], Loss: 0.1297
```
## 🧠 How It Works
🔹 Autograd Engine

Each Tensor keeps track of:

- Its data (CPU or GPU)

- Its gradient (.grad)

The operation that created it (if any)

A backward function that defines how gradients propagate

During backpropagation, the framework:

Traverses the computation graph in reverse.

Applies the chain rule to compute gradients.

Updates parameters via an optimizer.

🔹 GPU Support

Tensor operations are implemented in C++ and CUDA, allowing you to switch between CPU and GPU seamlessly:

x = norch.Tensor([1, 2, 3]).to("cuda")

⚡ Optimizer Example: SGD with Momentum

Example implementation of the SGD optimizer:
```bash
class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-1, momentum=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._cache = {'velocity': [p.zeros_like() for (_, _, p) in self.parameters]}

    def step(self):
        for i, (module, name, _) in enumerate(self.parameters):
            parameter = getattr(module, name)
            velocity = self._cache['velocity'][i]
            velocity = self.momentum * velocity - self.lr * parameter.grad
            updated_parameter = parameter + velocity
            setattr(module, name, updated_parameter)
            self._cache['velocity'][i] = velocity
            parameter.detach()
            velocity.detach()
```
### 🧰 Requirements

- Python ≥ 3.9

- CMake ≥ 3.18

- CUDA Toolkit ≥ 11.0

- A C++17-compatible compiler

# Install dependencies:
```bash
pip install -r requirements.txt
```
# 🧗‍♂️ Future Improvements

 - Add more activation functions (ReLU, Tanh, Softmax)

 - Support convolutional layers (Conv2D)

 - Add Adam optimizer

 - Expand GPU kernel support

 Implement dataset/dataloader utilities

# 📚 References

[PyTorch Internals](https://docs.pytorch.org/docs/stable/notes/autograd.html)

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

[Karpathy’s micrograd](https://github.com/karpathy/micrograd)

## 🏁 Conclusion

That’s it! 🎉
You’ve built a mini deep learning framework with:

- GPU acceleration

- Automatic differentiation

- Modular neural networks

- Real training capability

A great stepping stone to understanding how frameworks like PyTorch actually work under the hood.

# AUTHOR
- Simanga Mchunu