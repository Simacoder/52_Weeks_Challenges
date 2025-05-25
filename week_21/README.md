# 🩺 Diabetes Regression using Bagging and Boosting Trees

This project demonstrates how to apply **Ensemble Learning techniques**—specifically **Bagging** and **Boosting with Residual Correction**—to a regression problem using the classic **Diabetes dataset** from `scikit-learn`.

##  Project Highlights

- Implements **Bagging** by training multiple decision trees on bootstrap samples
- Implements **Boosting** using **residual fitting** to iteratively correct prediction errors
- Performs **manual grid search** over tree count, depth, and learning rate for Boosting
- Visualizes predicted vs actual target values for both methods
- Reports **Root Mean Squared Error (RMSE)** as the evaluation metric

---

##  Project Structure

```bash
.
├── ensemble_models.py   # Contains all core functions for Bagging & Boosting
├── main.py              # Example run: training, predicting, and visualizing results
├── README.md            # This file
```

### Dataset
The Diabetes dataset is a regression dataset included with scikit-learn, consisting of 442 samples and 10 baseline variables.

**Features include**:

- age, sex, bmi, bp (blood pressure)

- six blood serum measurements

The target is a quantitative measure of disease progression one year after baseline.

Models Implemented
1. **Bagging (Bootstrap Aggregating)**

- Trains n_trees decision trees on bootstrap samples

- Final prediction is the average of all individual predictions

2. **Boosting (Residual Correction)**

- Starts with a base tree predicting the target

- Subsequent trees fit to the residuals of previous predictions

- Final prediction is a sum of initial + scaled residual corrections

## Visualizations
- Both methods include a scatter plot showing actual vs predicted target values:

- Ideal predictions fall along the diagonal line

- Helps visually evaluate accuracy and bias/variance patterns

 **Evaluation**

Model performance is measured using Root Mean Squared Error (RMSE). The lower the RMSE, the better the model's predictions are on average.

# Requirements

- Python 3.7+

- NumPy

- Pandas

- scikit-learn

- Matplotlib

```bash
    pip install -r requirements.txt

```

## Future Ideas

- Add support for RandomForestRegressor and GradientBoostingRegressor from scikit-learn

- Use GridSearchCV for automated hyperparameter tuning

- Compare with XGBoost or LightGBM for performance benchmarking

## License

This project is open source and available under the MIT License.

## Author

Made with ❤️ by Simanga Mchunu
GitHub: @Simacoder