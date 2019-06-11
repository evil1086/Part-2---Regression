# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:34:32 2019

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT DATASETS
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# tranform categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()#LabelEncoder can be used to normalize labels.
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable
X = X[:, 1:]

#spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)

#visualizing the result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(y_train))
plt.show()