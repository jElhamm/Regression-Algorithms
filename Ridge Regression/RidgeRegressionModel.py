# This code implements a Ridge Regression model. Ridge Regression is a linear regression technique 
# that addresses the problem of multicollinearity (high correlation between input features) by introducing a regularization term.
# The regularization parameter, alpha, controls the amount of regularization applied.



import numpy as np


class RidgeRegressionModel:
    def __init__(self, X, y, alpha):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.weights = None

    def train(self): 
        X_with_bias = np.c_[self.X, np.ones(len(self.X))]               # Add bias column to the feature matrix
        identity_matrix = np.identity(X_with_bias.shape[1])             # Create an identity matrix
        # Calculate the weights using ridge regression formula
        self.weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias) + self.alpha * identity_matrix).dot(X_with_bias.T).dot(self.y)

    def predict(self, X):
        X_with_bias = np.c_[X, np.ones(len(X))]                         # Add bias column to the input matrix
        return X_with_bias.dot(self.weights)                            # Predict the output values