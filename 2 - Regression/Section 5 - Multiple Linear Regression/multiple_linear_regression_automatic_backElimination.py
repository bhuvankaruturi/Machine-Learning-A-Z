# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:18:59 2018

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

#Building the optimal model using Backward Elimination method 
#(One of the five available methods)
import statsmodels.formula.api as sm
#add a column of ones to our dataset to represent the variable X0 (for constant B0)
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

#automatic Backward Elimination Method
def backwardElimination(x, y, sl):
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y, exog = x).fit()
        maxPValue = max(regressor_OLS.pvalues).astype(float)
        if maxPValue > sl:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxPValue:
                    x = np.delete(x, j, 1)
        else:
            break
    regressor_OLS.summary()
    return x

#now find the the optimal variable set by inspecting the p-value
#for this example significance level SL = 0.05 (5%)
X_opt = backwardElimination(X, y, 0.05)

#split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fit multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict values using our model for test data
y_pred = regressor.predict(X_test)

#Perform the above steps for optimal set of variables X_opt
#split the dataset into training and test sets
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

#Fit multiple linear regression model to the training set
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, y_opt_train)

#predict values using our model for test data
y_opt_pred = regressor_opt.predict(X_opt_test)

#plot the predicted and actual test values for comparison
plot.scatter(np.arange(y_test.size), y_test, color = 'red')
plot.scatter(np.arange(y_test.size), y_pred, color = 'blue')
plot.scatter(np.arange(y_test.size), y_opt_pred, color = 'green')
plot.title('Predicted results for non-optimal, optimal variables and Actual profits')
plot.xlabel('Index')
plot.ylabel('Profit')
plot.show()