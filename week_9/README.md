# Polars Decision Tree Classifier

This project implements a **Decision Tree Classifier** using **Polars**, a high-performance DataFrame library for Python. The classifier supports both numerical and categorical features, and it efficiently builds decision trees for classification tasks.

## Features

- Decision Tree Classifier implemented with Polars
- Supports both **numerical** and **categorical** features
- Uses **target encoding** for categorical variables
- Supports **lazy evaluation** and **streaming** for large datasets
- Implements **entropy-based splitting** for decision tree construction
- Provides methods for **saving and loading** trained models using pickle
- Supports **batch predictions** as well as **single predictions**

## Installation

Ensure you have Python installed (>= 3.8). Then, install the required dependencies:

```bash
pip install polars
```

## Usage

### Import the Classifier

```python
from decision_tree import DecisionTreeClassifier
import polars as pl

# Sample data
data = pl.DataFrame({
    "feature_1": [1, 2, 3, 4, 5],
    "feature_2": ["A", "B", "A", "B", "A"],
    "target": [0, 1, 0, 1, 0]
})

# Initialize the model
model = DecisionTreeClassifier(max_depth=3, categorical_columns=["feature_2"])

# Train the model
model.fit(data, target_name="target")

# Make predictions
predictions = model.predict_many(data)
print(predictions)
```

### Saving and Loading the Model

```python
# Save the trained model
model.save_model("decision_tree.pkl")

# Load the model later
model.load_model("decision_tree.pkl")
```

### Predicting on New Data

```python
new_data = pl.DataFrame({
    "feature_1": [3, 1],
    "feature_2": ["A", "B"]
})

predictions = model.predict_many(new_data)
print(predictions)
```

## Project Structure

```
.
├── tree.py      # Implementation of the DecisionTreeClassifier
├── tests/                # Unit tests for the classifier
├── README.md             # Documentation
├── requirements.txt      # List of dependencies
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

