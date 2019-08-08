# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('test.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,  y_train, y_test = train_test_split(X, Y, test_size =0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

plt.scatter(x_test, y_pred, color='red')
#plt.plot(x_train,regression.predict(x_train), color = 'black')
plt.plot(x_train,regression.predict(x_train), color = 'blue')
plt.title('testing')
plt.xlabel('year')
plt.ylabel('salary')
plt.show()



# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'yellow')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

