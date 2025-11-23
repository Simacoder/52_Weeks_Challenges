import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# create a 20-point dataset from Fish Market dataset
data = [
    [11.52, 4.02, 242.0],
    [12.48, 4.30, 290.0],
    [12.38, 4.70, 340.0],
    [12.73, 4.46, 363.0],
    [12.44, 5.13, 430.0],
    [13.60, 4.93, 450.0],
    [14.18, 5.28, 500.0],
    [12.67, 4.69, 390.0],
    [14.00, 4.84, 450.0],
    [14.23, 4.96, 500.0],
    [14.26, 5.10, 475.0],
    [14.37, 4.81, 500.0],
    [13.76, 4.37, 500.0],
    [13.91, 5.07, 340.0],
    [14.95, 5.17, 600.0],
    [15.44, 5.58, 600.0],
    [14.86, 5.29, 700.0],
    [14.94, 5.20, 700.0],
    [15.63, 5.13, 610.0],
    [14.47, 5.73, 650.0],
]

df = pd.DataFrame(data, columns=["Height", "Width", "Weight"])
X = df[["Height", "Width"]]
y = df["Weight"]

model = LinearRegression().fit(X, y)
b0 = model.intercept_
b1, b2 = model.coef_

# Prepare grid for plane
height_range = np.linspace(df.Height.min(), df.Height.max(), 20)
width_range = np.linspace(df.Width.min(), df.Width.max(), 20)
H, W = np.meshgrid(height_range, width_range)
Z = b0 + b1 * H + b2 * W

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.Height, df.Width, df.Weight)
ax.plot_surface(H, W, Z, alpha=0.5)

ax.set_xlabel("Height")
ax.set_ylabel("Width")
ax.set_zlabel("Weight")

plt.show()


