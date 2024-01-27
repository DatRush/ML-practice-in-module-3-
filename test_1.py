import pandas as pd
import ssl
import numpy as np
from sklearn.metrics import mean_squared_error

ssl._create_default_https_context = ssl._create_unverified_context
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, sep=r'\s+', names=col_names)
data = data[data['B'] >= 50]

def linreg_linear(X, y):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

X = data.drop('MEDV', axis=1)
y = data['MEDV']

theta = linreg_linear(X, y)  

X = np.hstack([np.ones((X.shape[0], 1)), X])

y_pred = X.dot(theta)

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')

print_regression_metrics(y, y_pred)
