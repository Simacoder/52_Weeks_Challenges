# import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
# importing category_encoders for categorical encoding
import category_encoders as ce

# load the dataset

data = pd.read_csv('data/train.csv')

print(data.head())
print("Data shape:", data.shape)

# separate features and target variable

train_x = data.drop(columns=['Item_Outlet_Sales'])
train_y = data['Item_Outlet_Sales']


#define the class OutTypeEncoder
"""
 This class is used to encode the categorical variables in the dataset.
 It inherits from BaseEstimator and TransformerMixin to be used in a scikit-learn pipeline.
 custom encoding is applied to the 'Outlet_Type' column.
"""

class OutletTypeEnconder(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        x_dataset['outlet_grocery_store'] = (x_dataset['Outlet_Type'] == 'Grocery Store')*1
        x_dataset['outlet_supermarket_3'] = (x_dataset['Outlet_Type'] == 'Supermarket Type3')*1
        x_dataset['outlet_identifier_OUT027'] = (x_dataset['Outlet_Identifier'] == 'OUT027')*1

        return x_dataset


# create a pipeline for preprocessing
""" 
    Drop the columns that are not needed for the model.
    The columns 'Item_Identifier', 'Outlet_Establishment_Year', and 'Outlet_Type' are dropped.
"""

preprocessing_pipeline = ColumnTransformer(remainder='passthrough', transformers=[
('drop_columns', 'drop', ['Item_Identifier',                                                                     'Outlet_Identifier',
                            'Item_Fat_Content',
                             'Item_Type',
                             'Outlet_Identifier',
                             'Outlet_Size',
                             'Outlet_Location_Type',
                             'Outlet_Type'
                             ]),
                  ('impute_item_weight', SimpleImputer(strategy='mean'), ['Item_Weight']),
                 ('scale_data', StandardScaler(),['Item_MRP'])
])

# define the pipelin e
"""
    The pipeline consists of the following steps:
    1. Preprocessing: Drop unnecessary columns, impute missing values, and scale the data.
    2. Encode categorical variables: One-hot encode the categorical variables.      
"""

model_pipeline = Pipeline(steps=[('get_outlet_binary_columns', OutletTypeEnconder()),
('pre_processing', preprocessing_pipeline),
('random_forest_regressor', RandomForestRegressor(max_depth=10, random_state=2))
])

# fit the model witht the trainign data
model_pipeline.fit(train_x, train_y)

# predict the target variable
predictions = model_pipeline.predict(train_x)

print("Predictions:", predictions[:5])

print("For Test data")

# read the test data

test_data = pd.read_csv('data/test.csv')

# predict target variable for the test data
test_predictions = model_pipeline.predict(test_data)

print("Test Predictions:", test_predictions[:5])