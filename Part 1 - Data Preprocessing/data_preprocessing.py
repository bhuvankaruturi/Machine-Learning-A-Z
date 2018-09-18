import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

#import the csv using pandas 
#Note: use pandas while importing datasets

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#take care of missing values in the dataset
#Tip: hit ctrl + i to get info about a method
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])