# DIY AI and ML 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# load the dataset 
df = pd.read_csv('data/housing.csv')

X = df[['housing_median_age','total_rooms','total_bedrooms','median_income']]
y = df['median_house_value']

MLR = MultipleLinearRegression()
MLR.train_test_split(X = X,y=y,train_size = 0.8, random_state=42)
MLR.fit()
MLR.evaluate()