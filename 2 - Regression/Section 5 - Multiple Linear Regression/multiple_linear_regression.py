# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:42:33 2018

@author: bhuvan
"""

#Multiple Linear Regression
#sample model equation:
#y = B0 + B1 * X1 + B2 * X2 .... + Bn * Xn

#importing libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encode categorical variables (here the city variable)
#create dummy variables for categorical independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap
X = X[:, 1:]

#split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict values using our model for test data
y_pred = regressor.predict(X_test)

#plot the predicted and actual test values for comparison
plot.scatter(np.arange(y_test.size), y_test, color = 'red')
plot.scatter(np.arange(y_test.size), y_pred, color = 'blue')
plot.title('Predicted and Actual profits')
plot.xlabel('Index')
plot.ylabel('Profit')
plot.show()

#Building the optimal model using Backward Elimination method 
#(One of the five available methods)
import statsmodels.formula.api as sm
#add a column of ones to our dataset to represent the variable X0 (for constant B0)
#now find the the optimal variable set by inspecting the p-value
#for this example significance level SL = 0.05 (5%)
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Therefore only variable at index 3 has significant effect on profit value 
#Note: varaible at index 0 is just a constant and need not be considered as a independent variable