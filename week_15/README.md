#  DIY AI Series — Week 15: Multiple Linear Regression from Scratch

##  Introduction

With all the excitement and energy around AI, it’s easy to lose sight of the **foundational mathematics and technology** that make these models possible.

As professionals in the data and AI fields, one of the most rewarding ways to deepen your understanding is to **code models from the ground up** — without using machine learning libraries like `scikit-learn`, `TensorFlow`, or `PyTorch`.

This inspired the creation of the **DIY AI Series**, where we build machine learning models using pure Python and math. In this installment, we tackle **Multiple Linear Regression** — a fundamental algorithm used in countless real-world scenarios.

---

##  What is Multiple Linear Regression?

Multiple Linear Regression models the relationship between **two or more independent variables** and a **continuous dependent variable**.

###  Real-World Example

Predicting house prices using:
- Number of bedrooms
- Number of bathrooms
- Square footage
- Neighborhood score

---

## 🔍 Key Assumptions

1. **Linearity**  
   The relationship between features and target is linear. A one-unit change in a feature leads to a constant change in the outcome.

2. **No Multicollinearity**  
   Features should not be highly correlated with each other (e.g., bedrooms vs. bathrooms). Redundant variables distort coefficient estimation.

3. **Homoscedasticity**  
   Constant variance of residuals across all levels of independent variables. If prediction errors increase with the target value, this assumption is violated.

---

##  The Math Behind It

You may remember the simple linear equation:

\[
y = mx + b
\]

Multiple Linear Regression generalizes this to:

\[
y = B₀ + B₁x₁ + B₂x₂ + ... + Bₙxₙ + \epsilon
\]

Where:
- `y`: target variable
- `x₁...xₙ`: independent variables (features)
- `B₀`: intercept (bias term)
- `B₁...Bₙ`: coefficients (slopes)
- `ϵ`: error term

### Matrix Notation

We can rewrite it as:

\[
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
\]

To solve for coefficients \( \boldsymbol{\beta} \), we use the **Normal Equation**:

\[
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\]

This gives the optimal linear solution using linear algebra.

---

##  How It Works (Implementation Steps)

1. **Prepare the Design Matrix**  
   Add a column of 1s to the input features to account for the intercept term.

2. **Apply the Normal Equation**  
   Use matrix multiplication to compute coefficients.

3. **Predict Target Values**  
   \[
   \hat{y} = \mathbf{X} \boldsymbol{\beta}
   \]

4. **Optional: Evaluate**  
   Calculate R² score to assess model performance.

---

##  Getting Started

###  Requirements

- Python 3
- NumPy

###  Project Structure


### ▶ Run It

```bash
# Clone the repo
git clone https://github.com/Simacoder/52_Weeks_Challenges.git
cd 52_Weeks_Challenges/week_15

# Install numpy
pip install numpy

# Run the model
python multiple_linear_regression.py
```

# AUTHOR
- Simanga Mchunu
