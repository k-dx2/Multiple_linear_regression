import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

#reading the csv file
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
print(df.head())

# summarize the data
print(df.describe())


#selecting features for Linear regression and spliting it in test(30%) and train data set
X= df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y=df[['CO2EMISSIONS']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

#applying Linear regression between enginesize and CO2 emission
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)


# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
print ('\n')

#testing metrics
test_y_hat = regr.predict(X_test)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , y_test) )

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))


