# This class represents a decision tree regressor, used for predicting continuous values based on a set of features. 
# It performs regression by recursively constructing a decision tree based on the training data



import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        num_samples, num_features = X.shape
        # Check termination criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples <= 1:
            self.tree = np.mean(y)                                          # Set the tree value as the mean of y
            return

        best_mse = float("inf")
        best_feature_index = None
        best_threshold = None
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Separate the labels based on the current feature and threshold
                left_labels = y[X[:, feature_index] <= threshold]
                right_labels = y[X[:, feature_index] > threshold]
                mse = self._calculate_mse(left_labels, right_labels)
                if mse < best_mse:
                    best_mse = mse
                    best_feature_index = feature_index
                    best_threshold = threshold
        if best_threshold is None:                                          # Handle None case
            self.tree = np.mean(y)                                          # Set the tree value as the mean of y
            return

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask
        self.tree = {
            "feature_index": best_feature_index,
            "threshold": best_threshold,
            "left": DecisionTreeRegressor(max_depth=self.max_depth),        # Create left subtree
            "right": DecisionTreeRegressor(max_depth=self.max_depth),       # Create right subtree
        }
        self.tree["left"].fit(X[left_mask], y[left_mask], depth + 1)        # Fit left subtree
        self.tree["right"].fit(X[right_mask], y[right_mask], depth + 1)     # Fit right subtree

    # Calculate the mean squared error (MSE) by averaging the MSE of the left and right labels
    def _calculate_mse(self, left_labels, right_labels):
        mse_left = np.mean((left_labels - np.mean(left_labels)) ** 2)
        mse_right = np.mean((right_labels - np.mean(right_labels)) ** 2)
        mse = (len(left_labels) * mse_left + len(right_labels) * mse_right) / (
            len(left_labels) + len(right_labels)
        )
        return mse

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        if isinstance(self.tree, np.float64):                               # Leaf node reached, return the tree value
            return self.tree
        if sample[self.tree["feature_index"]] <= self.tree["threshold"]:    # Traverse left subtree
            return self.tree["left"]._predict_sample(sample)
        else:                                                               # Traverse right subtree
            return self.tree["right"]._predict_sample(sample)