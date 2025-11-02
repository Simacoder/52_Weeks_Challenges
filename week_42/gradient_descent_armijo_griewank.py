import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.ticker import MaxNLocator

# =========================================================
# 1️⃣ Armijo Line Search
# =========================================================
def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """
    Minimize over alpha the function f(x_k + α p_k)
    using the Armijo condition for sufficient decrease.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0 * pk)

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0 * pk)

    return alpha0, phi_a0

# =========================================================
# 2️⃣ Griewank Function & Gradient
# =========================================================
def Griewank(xs):
    """Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    sigma = np.dot(xs, xs) / 4000
    pi = np.prod(cos_terms)
    return 1 + sigma - pi

def GriewankGrad(xs):
    """First derivative of Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    pi_coefs = np.prod(cos_terms) / cos_terms
    sigma = 2 * xs / 4000
    pi = pi_coefs * np.sin(xs / sqrts) * (1 / sqrts)
    return sigma + pi

# =========================================================
# 3️⃣ Gradient Descent using Armijo Line Search
# =========================================================
def GradientDescent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    xk = init
    fk = f(xk)
    gfk = f_grad(xk)
    gfk_norm = np.linalg.norm(gfk)

    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]

    print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))

    while gfk_norm > tol and num_iter < max_iter:
        pk = -gfk
        alpha, fk = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = f_grad(xk)
        gfk_norm = np.linalg.norm(gfk)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, fk, xk, gfk_norm))

    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))

    return np.array(curve_x), np.array(curve_y)

# =========================================================
# 4️⃣ Visualization Helpers
# =========================================================
def plot_function_surface():
    """Plot 2D Griewank surface."""
    x = np.arange(-5, 5, 0.025)
    y = np.arange(-5, 5, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i, j in product(range(len(X)), range(len(Y))):
        Z[i][j] = Griewank(np.array([X[i][j], Y[i][j]]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('2D Griewank Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    plt.tight_layout()
    plt.show()
    return X, Y, Z

def plot(xs, ys, X, Y, Z):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle('Gradient Descent with Armijo Line Search')

    # Path on contour
    ax1.plot(xs[:, 0], xs[:, 1], '--o', color='orange')
    ax1.plot(xs[-1, 0], xs[-1, 1], 'ro')
    ax1.set(title='Path During Optimization', xlabel='x1', ylabel='x2')
    CS = ax1.contour(X, Y, Z, cmap='viridis')
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    # Function value per iteration
    ax2.plot(ys, '--o', color='orange')
    ax2.plot(len(ys) - 1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(title='Objective Value per Iteration',
            xlabel='Iterations', ylabel='Objective f(x)')
    ax2.legend(['Armijo Line Search'])

    plt.tight_layout()
    plt.show()

# =========================================================
# 5️⃣ Run Multiple Scenarios
# =========================================================
if __name__ == "__main__":
    X, Y, Z = plot_function_surface()

    initial_points = [
        np.array([0, 3]),
        np.array([1, 2]),
        np.array([2, 1]),
        np.array([1, 3]),
        np.array([2, 2]),
        np.array([3, 1])
    ]

    for i, x0 in enumerate(initial_points, 1):
        print(f"\n{'='*60}\nScenario {i}: x0 = {x0}\n{'='*60}")
        xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
        plot(xs, ys, X, Y, Z)
