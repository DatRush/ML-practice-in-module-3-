import pandas as pd
import ssl
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, sep=r'\s+', names=col_names)

class LinRegAlgebra():
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        self.theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    
    def predict(self, X):
        return X.dot(self.theta)

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')

def prepare_boston_data_new():
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    X_np = X.to_numpy()

    X_extended = np.hstack([X_np, np.sqrt(X_np[:, 5:6]), X_np[:, 6:7], X_np[:, 7:8] ** 3])

    X_normalized = (X_extended - X_extended.mean(axis=0)) / X_extended.std(axis=0)
    
    X_normalized = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])
    
    return X_normalized, y


def train_validate(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    
    linreg_alg = LinRegAlgebra()
    linreg_alg.fit(X_train, y_train)

    y_pred = linreg_alg.predict(X_valid)

    print_regression_metrics(y_valid, y_pred)
    
X, y = prepare_boston_data_new()

train_validate(X, y)