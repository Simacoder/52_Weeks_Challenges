import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression 

# create a 20-point dataset from Fish Market dataset
data = [
    [11.52,4.02, 242.0],
    [12.48,4.30, 290.0],
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
    [14.47, 5.73, 650.0]
]

# Create Dataframe 
df = pd.DataFrame(data, columns=["Height", "Width", "Weight"])
# Independent variables
X = df[["Height", "Width"]]

# target varieble
y = df["Weight"]

# Fit the model
model = LinearRegression().fit(X,y)

# Extract coefficients
b0 = model.intercept_ #B0
b1,b2 = model.coef_ #B1, B2

# print the results
print(f"Intercept (B0): {b0:.4f}")
print(f"Height slope (b1): {b1:.4f}")
print(f"Width slope (b2): {b2:.4f}")

