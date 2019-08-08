# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
path = '/Users/a-8525/Documents/expedia/projects/bex-sort-reporting/notebook/Participants_Data_Used_Cars/Data_Train.csv'
train_df = pd.read_csv(path)
path = '/Users/a-8525/Documents/expedia/projects/bex-sort-reporting/notebook/Participants_Data_Used_Cars/Data_Test.csv'
test_df = pd.read_csv(path)

# Pre processing
train_df['Mileage'] = train_df['Mileage'].str.extract('(\d+.\d+)')
train_df['New_Price'] = train_df['New_Price'].str.extract('(\d+.\d+)')
train_df['Power'] = train_df['Power'].str.extract('(\d+.\d+)')
train_df['Engine'] = train_df['Engine'].str.extract('(\d+)')


test_df['Mileage'] = test_df['Mileage'].str.extract('(\d+.\d+)')
test_df['New_Price'] = test_df['New_Price'].str.extract('(\d+.\d+)')
test_df['Power'] = test_df['Power'].str.extract('(\d+.\d+)')
test_df['Engine'] = test_df['Engine'].str.extract('(\d+)')


X = train_df.iloc[:, :-1].values
Y = train_df.iloc[:, -1].values


# Taking care of missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 9:11])
X[:,  9:11] = imputer.transform(X[:, 9:11])  




# Preprocessing
X = train_df.iloc[:, :-1].values
Y = train_df.iloc[:, -1].values

# Encoding categorical 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test result
y_pred = regressor.predict(X_test)

# Building optimal model using backward elimination
import statsmodels.formula.api as sm
X  =  np.append(arr = np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,3,4,4]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
"""


import math

x = np.array(range(0,90))
y = np.sin(x)
plt.plot(x,y)