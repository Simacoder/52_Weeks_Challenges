import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Detect whether we have a display (for interactive plotting)
import os
has_display = os.environ.get("DISPLAY") is not None

# Use a GUI backend if possible, else use Agg
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
            xs = [x0]
            ys = [func(x0)]
            x = x0

            # Run 10 steps of gradient descent
            for _ in range(10):
                grad = fprime(x)
                x = x - lr * grad
                xs.append(x)
                ys.append(func(x))

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
