import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Detect whether we have a display (for interactive plotting)
has_display = os.environ.get("DISPLAY") is not None

# Use GUI backend if possible, else use Agg
if has_display:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

# Objective function
def func(x):
    return x**2 - 2*x - 3

# Gradient of the function
def fprime(x):
    return 2*x - 2

# Simple gradient descent function
def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=1000):
    # initialize x, f(x), and -f'(x)
    xk = x0
    fk = func(xk)
    pk = -fprime(xk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    # take steps
    while abs(pk) > tol and num_iter < max_iter:
        # calculate new x, f(x), and -f'(x)
        xk = xk + alpha * pk
        fk = func(xk)
        pk = -fprime(xk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
    # print results
    if num_iter == max_iter:
        print(f'Gradient descent does not converge for alpha={alpha}, x0={x0}.')
    else:
        print(f'Solution found (alpha={alpha}, x0={x0}): y = {fk:.4f}, x = {xk:.4f}')
    
    return curve_x, curve_y

# Plot the function
def plotFunc(x0):
    x = np.linspace(-5, 7, 100)
    plt.plot(x, func(x))
    plt.plot(x0, func(x0), 'ro')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Objective Function')

# Plot gradient descent path
def plotPath(xs, ys, x0):
    plotFunc(x0)
    plt.plot(xs, ys, linestyle='--', marker='o', color='orange')
    plt.plot(xs[-1], ys[-1], 'ro')
    plt.title('Gradient Descent Path')

# Main block
if __name__ == "__main__":
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    x0_values = [-4, 6]

    fig, axes = plt.subplots(len(x0_values), len(learning_rates), figsize=(15, 6))

    for i, x0 in enumerate(x0_values):
        for j, lr in enumerate(learning_rates):
            xs, ys = GradientDescentSimple(func, fprime, x0, lr, tol=1e-5, max_iter=100)
            
            ax = axes[i, j]
            plt.sca(ax)
            plotPath(xs, ys, x0)
            ax.set_title(f'LR={lr}, x0={x0}')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Show or save based on environment
    if has_display:
        plt.show()
    else:
        output_file = "gradient_descent_paths.png"
        plt.savefig(output_file, dpi=300)
        print(f"Non-interactive environment detected. Plot saved as {output_file}")
