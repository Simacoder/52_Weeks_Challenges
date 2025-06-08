#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor




# reading the dataset 

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# checking the first few rows of training data
print(train_data.head())
print(test_data.head())

print("") # leaving a space

# check information of the dataset
print(train_data.info())
print(test_data.info())

# check if there is missing values
print(train_data.isna().sum())

print("") # leaving a space

# imputing misssing values in item weight and outlet_size
train_data.Item_Weight.fillna(train_data.Item_Weight.mean(), inplace=True)
train_data.Outlet_Size.fillna(train_data.Outlet_Size.mode()[0], inplace=True)

# check if there is missing values after imputation
print(train_data.isna().sum())
print("") # leaving a space
# check the unique values in the Outlet_Size column
print(train_data.Outlet_Size.unique())

# create an object of the categorical encoding oneHot
OHE = ce.OneHotEncoder(cols=[
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Identifier',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
    
],use_cat_names=True)

# Encode the categorical variables

train_data = OHE.fit_transform(train_data)
# check the first few rows of the transformed training data
print(train_data.head())

# create an object of the StandardScaler
scaler = StandardScaler()

# fit with Item_MRP
scaler.fit(np.array(train_data.Item_MRP).reshape(-1, 1))
# transform the Item_MRP column
train_data.Item_MRP = scaler.transform(np.array(train_data.Item_MRP).reshape(-1, 1))


# separate the independent and dependent(target) variables
train_X = train_data.drop(['Item_Identifier', 'Item_Outlet_Sales'], axis=1)
train_Y = train_data['Item_Outlet_Sales']

# randomly split the data
train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size = 0.25, random_state=0)

# sahpe of the training and testing data
print("Shape of the training data: ", train_x.shape)
print("Shape of the testing data: ", test_x.shape)
print("Shape of the training target: ", train_y.shape)
print("Shape of the testing target: ", test_y.shape)

# create an object of the LinearRegression model
model = LinearRegression()

# fit the model with training data
model.fit(train_x, train_y)

# predict the model with testing data
predictions_train = model.predict(train_x)
predictions_test = model.predict(test_x)

# calculate the mean squared error 
print('Mean Squared Error (Train): ', mean_squared_error(train_y, predictions_train)**0.5)
print('Mean Squared Error (Test): ', mean_squared_error(test_y, predictions_test)**0.5)

# create an object of the RandomForestRegressor model
model_rf = RandomForestRegressor(max_depth=10)

# fit the model with training data
model_rf.fit(train_x, train_y)

# predict the model with testing data

predictions_train_rf = model_rf.predict(train_x)
predictions_test_rf = model_rf.predict(test_x)

# calculate the mean squared error
print('For Random Forest Regressor:')
print('Mean Squared Error (Train RF): ', mean_squared_error(train_y, predictions_train_rf)**0.5)
print('Mean Squared Error (Test RF): ', mean_squared_error(test_y, predictions_test_rf)**0.5)

# plot the predictions vs actual values for Linear Regression
plt.figure(figsize=(10, 5))     
plt.scatter(test_y, predictions_test, color='blue', label='Predictions')
plt.scatter(test_y, test_y, color='red', label='Actual Values')
plt.title('Linear Regression Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()
# plot the predictions vs actual values for Random Forest Regressor
plt.figure(figsize=(10, 5))
plt.scatter(test_y, predictions_test_rf, color='blue', label='Predictions')
plt.scatter(test_y, test_y, color='red', label='Actual Values') 
plt.title('Random Forest Regressor Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

# plot 7 most importance features for Random Forest Regressor

plt.figure(figsize=(10, 5))
feat_importances = pd.Series(model_rf.feature_importances_, index=train_x.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.title('Feature Importance for Random Forest Regressor')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

#training the model with 7 most importance features
train_x_rf = train_x[['Item_MRP',
                       'Outlet_Type_Grocery Store',
                       'Item_Visibility',
                       'Outlet_Type_Supermarket Type3',
                       'Outlet_Identifier_OUT027',
                       'Outlet_Establishment_Year',
                       'Item_Weight'

        ]]

# test the model with 7 most importance features
test_x_rf = test_x[['Item_MRP',
                          'Outlet_Type_Grocery Store',
                          'Item_Visibility',
                          'Outlet_Type_Supermarket Type3',
                          'Outlet_Identifier_OUT027',
                          'Outlet_Establishment_Year',
                          'Item_Weight'
]]

# create an object of the RandomForestRegresor model

model_rf_7 = RandomForestRegressor(max_depth=10, random_state=2)

# fit the model with training data
model_rf_7.fit(train_x_rf, train_y)

# predict the model with testing data and training data
predictions_train_rf_7 = model_rf_7.predict(train_x_rf)
predictions_test_rf_7 = model_rf_7.predict(test_x_rf)

# root mean squared error for training and testing data

print("fine tuning the model with 7 most importance features")
print('Mean Squared Error (Train RF 7 Features):', mean_squared_error(train_y, predictions_train_rf_7)**0.5)
print('Mean Squared Error (Test RF 7 Features):', mean_squared_error(test_y, predictions_test_rf_7)**0.5)