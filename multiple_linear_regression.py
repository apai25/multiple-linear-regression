# Importing Numpy and Pandas
import numpy as np 
import pandas as pd     

# Importing the dataset and creating x and y datasets
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, -1].values

# Categorically encoding "State" predictor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
encoder = OneHotEncoder(drop='first', dtype=int)
ct = ColumnTransformer([('categorical_encoding', encoder, [3])], remainder='passthrough')
x = ct.fit_transform(x)

# Creating training and tests sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0)

# Creating, training, and testing our linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

predictions = regressor.predict(x_test)

# Calculating the adjusted r squared metric to see our results
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, predictions)
N = len(x_test)
p = 5
adjusted_r_squared = 1 - (((1 - (r_squared ** 2)) * (N - 1)) / (N - p - 1))
print(f'The adjusted R score of our model is: {adjusted_r_squared}')