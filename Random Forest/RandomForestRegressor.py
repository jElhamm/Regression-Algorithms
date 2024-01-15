# Implementation of RandomForestRegressor that uses an ensemble of decision trees for regression


import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.estimators = []

        # Create a forest of decision tree estimators
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)          # Random sampling with replacement
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)                  # Create a decision tree regressor
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)                                            # Add the tree to the ensemble

    # Make predictions by aggregating predictions from all trees in the ensemble
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.mean(predictions, axis=0)