# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:54:21 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##reading the dataset Data,csv
dataset = pd. read_csv ('Position_Salaries.csv')
X=dataset.iloc [:,1:-1].values
Y=dataset.iloc [:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,Y)
y_pred=regressor.predict([[6.5]])
print("the estimated salary is=",y_pred)
