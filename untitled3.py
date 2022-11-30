# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:27:47 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##reading the dataset Data,csv
dataset = pd. read_csv ('Position_Salaries.csv')
X=dataset.iloc [:,1:-1].values
Y=dataset.iloc [:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)
Y_pred = regressor.predict([[6.5]])
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='Blue')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('Postion vs salary(truth or bluff)')
plt.xlabel('position')
plt.ylabel('salary')
plt.legend('DTR', 'origial')
plt.show()