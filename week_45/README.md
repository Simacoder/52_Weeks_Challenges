# Multiple Linear Regression Explained Simply

## The Math Behind Fitting a Plane Instead of a Line

Photo by Ortega Castro 🇲🇽 on Unsplash

---

## Overview

This guide explores **multiple linear regression** from first principles, focusing on the mathematical foundations rather than just applying algorithms. While simple linear regression uses one independent variable to predict one target, multiple linear regression extends this to two or more independent variables—requiring us to fit a plane instead of a line.

---

## Dataset: Fish Market

To illustrate the concepts, we use the Fish Market dataset, which includes physical attributes of fish:

- **Species**: The type of fish (e.g., Bream, Roach, Pike)
- **Weight**: The weight of the fish in grams *(target variable)*
- **Length1, Length2, Length3**: Various length measurements in cm
- **Height**: The height of the fish in cm
- **Width**: The diagonal width of the fish body in cm

For simplicity and visualization, this guide uses **two independent variables** (Height and Width) and a **20-point sample** from the full dataset.

---

## Quick Start: Fitting a Model in Python

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 20-point sample data from Fish Market dataset
data = [
    [11.52, 4.02, 242.0],
    [12.48, 4.31, 290.0],
    [12.38, 4.70, 340.0],
    # ... (additional 17 data points)
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Height", "Width", "Weight"])

# Independent variables (Height and Width)
X = df[["Height", "Width"]]

# Target variable (Weight)
y = df["Weight"]

# Fit the model
model = LinearRegression().fit(X, y)

# Extract coefficients
b0 = model.intercept_  # β₀
b1, b2 = model.coef_   # β₁ (Height), β₂ (Width)

print(f"Intercept (β₀): {b0:.4f}")
print(f"Height slope (β₁): {b1:.4f}")
print(f"Width slope (β₂): {b2:.4f}")
```

**Results:**
- Intercept (β₀): -1005.2810
- Height slope (β₁): 78.1404
- Width slope (β₂): 82.0572

---

## Understanding the Geometry

### From Lines to Planes

In **simple linear regression**, we fit a line through 2D data. The equation is:

$$y = \beta_0 + \beta_1 x_1$$

In **multiple linear regression** with two features, we fit a plane through 3D data:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

Where:
- $\hat{y}$: The predicted value of the dependent (target) variable
- $\beta_0$: The intercept (the value of y when all x's are 0)
- $\beta_1$: The coefficient (or slope) for feature $x_1$
- $\beta_2$: The coefficient for feature $x_2$
- $x_1, x_2$: The independent variables (features)

---

## The Residuals and Sum of Squared Residuals

For any point $i$ in our dataset, we can compute:

**Predicted value:** $\hat{y}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2}$

**Residual:** $\text{Residual}_i = y_i - \hat{y}_i$

The **Sum of Squared Residuals (SSR)** measures total prediction error:

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{i1} - \beta_2 x_{i2})^2$$

Squaring ensures all errors contribute positively and gives more weight to larger deviations.

---

## The Mathematical Derivation

### Why Calculus?

The goal is to find the values of $\beta_0$, $\beta_1$, and $\beta_2$ that **minimize** the SSR. This is where differentiation becomes essential.

For curves with multiple variables, we use **partial differentiation**—differentiating each variable separately while treating others as constants.

### Setting Up the Equations

We define the loss function:

$$L(\beta_0, \beta_1, \beta_2) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{i1} - \beta_2 x_{i2})^2$$

At the minimum, all partial derivatives equal zero:

$$\frac{\partial L}{\partial \beta_0} = 0, \quad \frac{\partial L}{\partial \beta_1} = 0, \quad \frac{\partial L}{\partial \beta_2} = 0$$

### Partial Derivatives

**With respect to β₀:**

$$\frac{\partial L}{\partial \beta_0} = -2 \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{i1} - \beta_2 x_{i2}) = 0$$

This simplifies to:

$$\beta_0 = \bar{y} - \beta_1 \bar{x}_1 - \beta_2 \bar{x}_2$$

**With respect to β₁ and β₂:**

Similar partial differentiation yields two more equations that, when solved together using Cramer's Rule, give:

$$\beta_1 = \frac{(\sum x_{i2}^2)(\sum x_{i1} y_i) - (\sum x_{i1} x_{i2})(\sum x_{i2} y_i)}{(\sum x_{i1}^2)(\sum x_{i2}^2) - (\sum x_{i1} x_{i2})^2}$$

$$\beta_2 = \frac{(\sum x_{i1}^2)(\sum x_{i2} y_i) - (\sum x_{i1} x_{i2})(\sum x_{i1} y_i)}{(\sum x_{i1}^2)(\sum x_{i2}^2) - (\sum x_{i1} x_{i2})^2}$$

---

## Centering the Data

**Centering** means subtracting the mean from each variable:

$$x'_{i1} = x_{i1} - \bar{x}_1, \quad x'_{i2} = x_{i2} - \bar{x}_2, \quad y'_i = y_i - \bar{y}$$

### Benefits of Centering:

- Simplifies formulas by eliminating extra terms
- Ensures the mean of all variables is zero
- Improves numerical stability
- Makes the intercept easier to calculate: $\beta_0 = \bar{y}$ (for centered data)

### Example:

For a small dataset with 3 observations:

| i | Original x₁ | Original x₂ | Original y | Centered x'₁ | Centered x'₂ | Centered y' |
|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 10 | -2 | -2 | -4 |
| 2 | 4 | 5 | 14 | 0 | 0 | 0 |
| 3 | 6 | 7 | 18 | +2 | +2 | +4 |

After centering: $\sum x'_{i1} = 0$, $\sum x'_{i2} = 0$, $\sum y'_i = 0$

---

## Computing Coefficients for Our Sample Data

### Step 1: Compute Means (Original Data)

$$\bar{x}_1 = 13.841, \quad \bar{x}_2 = 4.9385, \quad \bar{y} = 481.5$$

### Step 2: Center the Data

Subtract means from all observations.

### Step 3: Compute Centered Summations

$$\sum x'_{i1} y'_i = 2465.60, \quad \sum x'_{i2} y'_i = 816.57$$

$$\sum (x'_{i1})^2 = 24.3876, \quad \sum (x'_{i2})^2 = 3.4531, \quad \sum x'_{i1} x'_{i2} = 6.8238$$

### Step 4: Compute Shared Denominator

$$\Delta = (24.3876)(3.4531) - (6.8238)^2 = 37.6470$$

### Step 5: Compute Slopes

$$\beta_1 = \frac{(3.4531)(2465.60) - (6.8238)(816.57)}{37.6470} = \frac{2940.99}{37.6470} = 78.14$$

$$\beta_2 = \frac{(24.3876)(816.57) - (6.8238)(2465.60)}{37.6470} = \frac{3089.79}{37.6470} = 82.06$$

### Step 6: Compute Intercept

$$\beta_0 = \bar{y} - \beta_1 \bar{x}_1 - \beta_2 \bar{x}_2$$

$$\beta_0 = 481.5 - (78.14)(13.841) - (82.06)(4.9385) = -1005.28$$

---

## Final Regression Equation

$$\hat{y}_i = -1005.28 + 78.14 \cdot x_{i1} + 82.06 \cdot x_{i2}$$

This is how we derive the coefficients that Python's `LinearRegression` computes behind the scenes!

---

## Key Takeaways

1. **Multiple linear regression** extends simple linear regression by fitting a plane (or hyperplane) through multi-dimensional data.

2. **The goal** is to find coefficients that minimize the Sum of Squared Residuals.

3. **Calculus is essential**: We use partial differentiation to find the point where the gradient is zero—the minimum of the cost function.

4. **Three unknowns** ($\beta_0$, $\beta_1$, $\beta_2$) require solving a system of three equations.

5. **Data centering** simplifies calculations and improves numerical stability.

6. **The final equation** is a direct result of mathematical optimization, not trial-and-error.

---

## What's Next?

Part 2 of this series will cover model evaluation, interpreting coefficients, and handling more than two features. Stay tuned!

---

## References

- Fish Market Dataset
- Linear Algebra and Calculus fundamentals
- Multiple Linear Regression theory

---

**Author:** Simanga Mchunu 