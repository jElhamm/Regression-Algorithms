# This class represents a decision tree regressor, used for predicting continuous values based on a set of features. 
# It performs regression by recursively constructing a decision tree based on the training data



import numpy as np
from TreeNode import TreeNode


class DecisionTreeRegressor:
    def __init__(self, X, y, max_depth=None):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.root = None

    def train(self):
        self.root = self._build_tree(self.X, self.y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        # Check for termination criteria and return a leaf node with the calculated leaf value
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples <= 1:
            leaf_value = self._get_leaf_value(y)
            return TreeNode(value=leaf_value)
        
        # Find the best split point based on the lowest mean squared error (MSE)
        best_feature_index, best_threshold = self._find_best_split(X, y)

        # Create masks to separate the data into left and right subtrees
        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=best_feature_index, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_feature_index = None
        best_threshold = None
        # Iterate over all features and thresholds to find the best split based on the lowest MSE
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Separate the labels based on the current feature and threshold
                left_labels = y[X[:, feature_index] <= threshold]           
                right_labels = y[X[:, feature_index] > threshold]
                # Calculate the MSE for the current split
                mse = self._calculate_mse(left_labels, right_labels)

                # Update the best split if the current MSE is lower
                if mse < best_mse:
                    best_mse = mse
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_mse(self, left_labels, right_labels):
        # Calculate the mean squared error (MSE) by averaging the MSE of the left and right labels
        mse_left = np.mean((left_labels - np.mean(left_labels)) ** 2)
        mse_right = np.mean((right_labels - np.mean(right_labels)) ** 2)
        mse = (len(left_labels) * mse_left + len(right_labels) * mse_right) / (len(left_labels) + len(right_labels))
        return mse

    def _get_leaf_value(self, y):
        # Return the mean of the labels as the leaf valu
        return np.mean(y)           

    def predict(self, X):
        # Predict the output for each sample in X by traversing the decision tree
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.root
        while node.left:            # Traverse the decision tree starting from the root node until reaching a leaf node
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value