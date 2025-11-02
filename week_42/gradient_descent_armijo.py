import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Detect display availability
has_display = os.environ.get("DISPLAY") is not None
matplotlib.use("TkAgg" if has_display else "Agg")

# Objective function and derivative
def func(x):
    return x**2 - 2*x - 3

def fprime(x):
    return 2*x - 2

# --------------------------------------------------
# Armijo Line Search
# --------------------------------------------------
def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Perform Armijo backtracking line search."""
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0 * pk)
    
    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 *= rho
        phi_a0 = f(xk + alpha0 * pk)
        
    return alpha0, phi_a0

# --------------------------------------------------
# Gradient Descent with Armijo Line Search
# --------------------------------------------------
def GradientDescentArmijo(f, grad, x0, tol=1e-5, max_iter=1000, alpha0=1.0):
    xk = x0
    fk = f(xk)
    gfk = grad(xk)
    pk = -gfk

    xs = [xk]
    ys = [fk]

    iter_count = 0

    while np.linalg.norm(gfk) > tol and iter_count < max_iter:
        alpha, _ = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0)
        xk = xk + alpha * pk
        fk = f(xk)
        gfk = grad(xk)
        pk = -gfk

        xs.append(xk)
        ys.append(fk)
        iter_count += 1

    if iter_count == max_iter:
        print("Gradient descent did not converge.")
    else:
        print(f"Solution found: x = {xk:.4f}, f(x) = {fk:.4f} after {iter_count} iterations.")

    return np.array(xs), np.array(ys)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
def plotFunc(ax):
    x = np.linspace(-5, 7, 200)
    ax.plot(x, func(x), label='$f(x)$', color='steelblue')
    ax.axvline(1, color='green', linestyle='--', label='True Min ($x=1$)')
    ax.scatter(1, func(1), color='green', s=50)

def plotPath(xs, ys, x0):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotFunc(ax)
    ax.plot(xs, ys, linestyle='--', marker='o', color='orange', label='GD Path')
    ax.scatter(xs[0], ys[0], color='red', s=50, label='Start')
    ax.scatter(xs[-1], ys[-1], color='purple', s=50, label='End')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend()
    ax.set_title('Gradient Descent with Armijo Line Search')
    plt.tight_layout()

    if has_display:
        plt.show()
    else:
        plt.savefig("gradient_descent_armijo.png", dpi=300)
        print("Non-interactive environment: plot saved as gradient_descent_armijo.png")

# --------------------------------------------------
# Run Example
# --------------------------------------------------
if __name__ == "__main__":
    x0 = -4
    xs, ys = GradientDescentArmijo(func, fprime, x0)
    plotPath(xs, ys, x0)
