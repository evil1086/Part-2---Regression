# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:01:29 2019

@author: user
"""
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT DATASETS
dataset = pd.read_csv('Salary_Data.csv')

#salary
X = dataset.iloc[:, :-1].values

#Years of experience
y = dataset.iloc[:, 1].values

#spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#implementing linear regresion
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predecting the salary
y_pred = regressor.predict(X_test)

#visualizing training testset
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Training- testset')
plt.show()

#visualizing test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Testset')
plt.show()
