import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Detect whether we have a display (for interactive plotting)
has_display = os.environ.get("DISPLAY") is not None
matplotlib.use("TkAgg" if has_display else "Agg")

# Objective function and derivative
def func(x):
    return x**2 - 2*x - 3

def fprime(x):
    return 2*x - 2

# Gradient Descent Implementation
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
        print(f"Gradient descent does not converge (alpha={alpha}).")
    else:
        print(f"Solution found:  y = {fk:.4f},  x = {xk:.4f}  (alpha={alpha})")

    return curve_x, curve_y

# Plot the function
def plotFunc(ax, x0):
    x = np.linspace(-5, 7, 200)
    ax.plot(x, func(x), label='$f(x)$', color='steelblue')
    ax.plot(x0, func(x0), 'ro', label='Start ($x_0$)')
    # Highlight analytical solution x=1
    ax.axvline(1, color='green', linestyle='--', label='True Min ($x=1$)')
    ax.scatter(1, func(1), color='green', s=50)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend(loc='best')

# Plot the gradient descent path
def plotPath(ax, xs, ys, x0, alpha):
    plotFunc(ax, x0)
    ax.plot(xs, ys, linestyle='--', marker='o', color='orange', label='Path')
    ax.plot(xs[-1], ys[-1], 'ro', label='Final Point')
    ax.set_title(f'α = {alpha}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

# === Main Script ===
if __name__ == "__main__":
    x0 = -4
    scenarios = [0.1, 0.9, 1e-4, 1.01]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, alpha in enumerate(scenarios):
        # Limit iterations for divergent case
        max_iter = 8 if alpha > 1 else 1000
        xs, ys = GradientDescentSimple(func, fprime, x0, alpha, max_iter=max_iter)
        plotPath(axes[i], xs, ys, x0, alpha)

    fig.suptitle("Gradient Descent Behavior for Different Learning Rates", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Display or save the figure
    if has_display:
        plt.show()
    else:
        output_file = "gradient_descent_scenarios.png"
        plt.savefig(output_file, dpi=300)
        print(f"Non-interactive environment detected. Plot saved as {output_file}")
