# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:46:39 2018

@author: bhuvan
"""

#Polynomial regression
#used when there is a exponential relation between independent and dependent variables

#importing libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#reading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the data with linear regression
#Linear regression is merely being done
#to compare the results with the ones
#produced by polynomial regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#choosing which regression to use is a decision to made by the programmer based on data

#Fitting the data with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the linear regression results
plot.scatter(X, y, color = 'red')
plot.plot(X, lin_reg.predict(X), color = 'blue')


#Visualizing the polynomial regression results
#increasing the resolution(smoothness) of the curve by providing intermediate values of X
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plot.scatter(X, y, color = 'red')
plot.plot(X_grid, lin_reg_2.predict(poly_feat.fit_transform(X_grid)), color = 'green')
plot.title('Linear regression VS Polynomial Linear regression')
plot.xlabel('Position level')
plot.ylabel('Salary')
plot.show()

#predicting result with linear regression
lin_reg.predict(6.5) #result - 330378

#predicting result with polynomial regression
lin_reg_2.predict(poly_feat.transform(6.5)) #result - 158862