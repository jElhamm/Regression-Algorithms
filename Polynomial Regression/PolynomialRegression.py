# This code defines a Polynomial Regression Model class that can be used to perform polynomial regression.
# The __init__ method initializes the model with the input data (X), output data (y), and the degree of the polynomial.
# The train method fits the model to the training data by calculating the weights of the polynomial regression.
# The predict method can be used to make predictions on new input data using the learned weights.
# The _add_polynomial_features method is a helper function that adds polynomial features to the 
# input data by raising it to different powers up to the specified degree.



import numpy as np


class PolynomialRegressionModel:
    def __init__(self, X, y, degree):
        self.X = X
        self.y = y
        self.degree = degree
        self.weights = None

    def train(self):
        X_with_bias = self._add_polynomial_features(self.X, self.degree)                 # Add polynomial features to X
        self.weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(self.y)  # Calculate weights

    def predict(self, X):
        X_with_bias = self._add_polynomial_features(X, self.degree)                      # Add polynomial features to X
        return X_with_bias.dot(self.weights)                                             # Predict using weights

    def _add_polynomial_features(self, X, degree):
        n_samples = X.shape[0]
        X_with_bias = np.ones((n_samples, 1))                                            # Start with a column of ones (bias)
        for d in range(1, degree + 1):
            X_with_bias = np.c_[X_with_bias, X ** d]                                     # Add polynomial features
        return X_with_bias