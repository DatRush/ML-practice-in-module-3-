import pandas as pd
import ssl
import numpy as np
from sklearn.metrics import mean_squared_error

ssl._create_default_https_context = ssl._create_unverified_context
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, sep=r'\s+', names=col_names)

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f'MSE = {mse:.2f}, RMSE = {rmse:.2f}')
    
class LinRegAlgebra():
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        self.theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    
    def predict(self, X):
        return X.dot(self.theta)
    
class RegOptimizer():
    def __init__(self, alpha, n_iters):
        self.theta = None
        self._alpha = alpha
        self._n_iters = n_iters
    
    def gradient_step(self, theta, theta_grad):
        return theta - self._alpha * theta_grad
    
    def grad_func(self, X, y, theta):
        raise NotImplementedError()

    def optimize(self, X, y, start_theta, tol=0.01):
        theta = start_theta.copy()
        for i in range(self._n_iters):
            theta_grad = self.grad_func(X, y, theta)
            if max(abs(theta_grad)) < tol:
                print(f"Остановка на итерации {i}")
                break
            theta = self.gradient_step(theta, theta_grad)
        return theta, i
    
    def fit(self, X, y):
        m = X.shape[1]
        start_theta = np.ones(m)
        self.theta, _ = self.optimize(X, y, start_theta, tol=0.01)
        
    def predict(self, X):
        raise NotImplementedError()
    
class LinReg(RegOptimizer):
    def grad_func(self, X, y, theta):
        n = X.shape[0]
        grad = 1. / n * X.transpose().dot(X.dot(theta) - y)

        return grad
    
    def predict(self, X):
        if self.theta is None:
            raise Exception('You should train the model first')

        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[1] + 1 == len(self.theta):
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        y_pred = X.dot(self.theta)
        
        return y_pred
    
    
X = data.drop('MEDV', axis=1)
y = data['MEDV']

X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
X_normalized = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])


linreg_crit = LinReg(0.2,1000)
linreg_crit.fit(X_normalized, y)
y_pred = linreg_crit.predict(X_normalized)

# Посчитать значение ошибок MSE и RMSE для тренировочных данных
print_regression_metrics(y, y_pred)


