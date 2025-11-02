# 🧠 Gradient Descent Algorithm from Scratch

This repository implements **Gradient Descent** — one of the fundamental optimization algorithms — entirely **from scratch in Python**, including both:
- Constant learning rate (fixed step size)
- Adaptive learning rate using **Armijo Line Search**

The goal is to deeply understand how optimization works under the hood before relying on libraries like PyTorch or TensorFlow.

---

## 📚 Table of Contents
- [What is Optimization?](#what-is-optimization)
- [Gradient Descent (the Easy Way)](#gradient-descent-the-easy-way)
- [Armijo Line Search](#armijo-line-search)
- [Gradient Descent (the Hard Way)](#gradient-descent-the-hard-way)
- [Conclusion](#conclusion)
- [How to Run the Code](#how-to-run-the-code)
- [Project Files](#project-files)

---

## ⚙️ What is Optimization?

If you’ve been studying machine learning long enough, you’ve probably heard terms such as **SGD** or **Adam**. They are two of many *optimization algorithms*.  
Optimization algorithms are the **heart of machine learning**, responsible for adjusting model parameters to minimize (or maximize) an objective function — e.g., loss, error, or cost.

Optimization isn’t new — people and nature have been doing it forever:

- 📈 **Investors** aim to balance risk and return.
- 🏭 **Manufacturers** strive to maximize efficiency.
- ⚙️ **Engineers** tune parameters for performance.
- 🌿 Even **nature** optimizes — molecules arrange to minimize potential energy, and light follows the path of least time.

Formally, optimization means finding the variable **x** that minimizes an objective function **f(x)**, possibly under some constraints **cᵢ**:

\[
\min_{x} \ f(x) \quad \text{s.t.} \quad c_i(x) = 0, \quad c_j(x) \le 0
\]

But in our examples, we’ll keep it simple — **unconstrained optimization**.

---

## 🧮 Gradient Descent (the Easy Way)

To find a minimum, we can use calculus. The **gradient** tells us the direction of steepest ascent.  
So, to *minimize* a function, we move **opposite** to the gradient.

Imagine a ball rolling down a hill — at every point, it moves downhill, following gravity, until it reaches a valley (a local minimum).

At iteration **k**, we compute:
\[
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
\]

Where:
- \( \alpha_k \) → learning rate (step size)
- \( \nabla f(x_k) \) → gradient of f at xₖ

A small \( \alpha_k \) means slow progress,  
a large \( \alpha_k \) may cause divergence.

### 🔹 Example
We start from an initial guess and take steps proportional to the negative gradient:

```python
import numpy as np
import matplotlib.pyplot as plt

def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=1000):
    xk = x0
    fk = func(xk)
    pk = -fprime(xk)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]

    while abs(pk) > tol and num_iter < max_iter:
        xk = xk + alpha * pk
        fk = func(xk)
        pk = -fprime(xk)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)

    if num_iter == max_iter:
        print('Gradient descent does not converge.')
    else:
        print(f'Solution found: y = {fk:.4f}, x = {xk:.4f}')
    return curve_x, curve_y
```

Different learning rates yield drastically different results:

- ✅ α = 0.1 → converges perfectly

- ⚠️ α = 1e-4 → too slow

- ⚠️ α = 0.9 → oscillates

- ❌ α = 1.01 → diverges

# 🔍 Armijo Line Search

Instead of using a constant learning rate, we can determine the best step size αₖ dynamically for each iteration.

The Armijo condition ensures that we choose αₖ that gives a sufficient decrease in f(x):	​


If the condition isn’t met, we shrink αₖ by multiplying with ρ (e.g., 0.5) until it holds.

🔹 Implementation
```bash
def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0 * pk)

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0 * pk)
    
    return alpha0, phi_a0
```

This approach automatically adjusts step size, stabilizing convergence.

# 💡 Gradient Descent (the Hard Way)

We now combine everything to build Gradient Descent with Armijo Line Search — tested on the Griewank function, a non-convex function used in optimization benchmarks.


# 🔹 Features

Gradient computed analytically

Adaptive step size via Armijo

Visualizes convergence and function surface

🔹 Run Example
```bash
python gradient_descent_armijo_griewank.py
```

You’ll see:

3D Griewank surface plot

Path of optimization on contour

Objective value per iteration

# 🏁 Conclusion

We’ve built gradient descent from scratch — both with:

- Constant learning rate

- Adaptive learning rate using Armijo Line Search

# Key insights:

Step size is crucial for convergence.

- Line search offers a principled way to adapt α dynamically.

- Optimization principles power everything from simple regression to deep learning.

- Next steps: explore Conjugate Gradient or Newton’s Method for even faster convergence!

# 🧩 Project Files

File	Description

sgd1.py	Gradient Descent with constant learning rate

gradient_descent_armijo_griewank.py	Gradient Descent with Armijo Line Search

README.md	Project documentation

## 🚀 How to Run the Code

**Clone this repository:**

```bash
git clone https://github.com/yourusername/52_weeeks_challenges.git
cd 52_weeks_challenges
cd week_42
```

# Install dependencies:
```bash
pip install numpy matplotlib


Run each script:

python sgd1.py
python gradient_descent_armijo_griewank.py
```
# 💬 Author
- Simanga Mchunu

Created with ❤️ to demystify optimization algorithms and understand how learning truly happens from first principles.