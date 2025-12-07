# PyTorch Introduction – Building Your First Linear Model

This project is a hands-on introduction to PyTorch, where we build and train a **linear regression model** from scratch. It demonstrates the basics of PyTorch modules, forward passes, training loops, and optimization using SGD and Adam optimizers.

---

## Project Overview

Linear regression is one of the simplest machine learning models. In this project, we implement a custom PyTorch module to perform linear regression. The model is trained to minimize **mean absolute error (MAE)** on the dataset, and we compare the performance of different optimizers.

Key features:

- Custom PyTorch `LinearRegressionModel` class
- Forward pass, weight initialization, and bias
- Training using **SGD** and **Adam** optimizers
- Evaluation on a test set
- Visualizing train and test loss curves
- Comparison between SGD and Adam performance

---

## Files

- `linear_model.py` – Contains the implementation of `LinearRegressionModel`
- `train.py` – Script to load data, initialize the model, train, and plot results
- `README.md` – Project description and instructions

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install torch matplotlib numpy
```

# How to Run

# Prepare your data:
```bash
# X_train, X_test, y_train, y_test should be torch.Tensor
# For example:
X_train = torch.tensor(train_features, dtype=torch.float)
y_train = torch.tensor(train_targets, dtype=torch.float)
X_test = torch.tensor(test_features, dtype=torch.float)
y_test = torch.tensor(test_targets, dtype=torch.float)
```

Initialize and train the models:
```bash
input_dim = X_train.shape[1]
output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

sgd_model = LinearRegressionModel("SGD", input_dim, output_dim)
adam_model = LinearRegressionModel("Adam", input_dim, output_dim)

sgd_model.trainModel(500, X_train, X_test, y_train, y_test, lr=0.001)
adam_model.trainModel(200, X_train, X_test, y_train, y_test, lr=0.001)
```

# Plot the loss curves:
```bash
sgd_model.plotLoss()
adam_model.plotLoss()
```
# Compare SGD and Adam in one plot
- plotOptimizerComparison(sgd_model, adam_model)

# Results

- The model outputs train and test MAE loss at every 10th epoch.

- Plots show how quickly the model converges for different optimizers.

- Adam usually converges faster than SGD for the same learning rate.

[!Notes]

> The model automatically adjusts to the number of input features and output targets.

> Works for single-output or multi-output regression.

> Loss functions, optimizer choice, learning rate, and epochs can be easily modified.

# Author

- Simanga Mchunu

# References

[PyTorch Official Documentation](https://docs.pytorch.org/docs/stable/index.html)

[Deep Learning with PyTorch – Book](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://isip.piconepress.com/courses/temple/ece_4822/resources/books/Deep-Learning-with-PyTorch.pdf)