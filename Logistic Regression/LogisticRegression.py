# This program implements the Logistic Regression algorithm using the 'Singleton design pattern'.
# Logistic regression is a popular machine learning algorithm used for binary classification tasks.
# The Singleton design pattern ensures that only one instance of the LogisticRegression class is created,
# allowing global access to the instance throughout the program.



import numpy as np

class LogisticRegression:
    _instance = None
  
    def __init__(self):
        if LogisticRegression._instance is not None:
            raise Exception("LogisticRegression class is a singleton. Use get_instance() method to get the instance.")
        else:
            LogisticRegression._instance = self

        self.weights = None
        self.bias = None

    @staticmethod
    def get_instance():
        if LogisticRegression._instance is None:
            LogisticRegression()
        return LogisticRegression._instance
 
    @staticmethod
    def normalizeFeatures(features):
        mean = np.mean(features, axis=0)                            # Calculate the mean value of each feature
        std = np.std(features, axis=0)                              # Calculate the standard deviation of each feature
        # Normalize the features by subtracting the mean and dividing by the standard deviation
        featuresNormalized = (features - mean) / std
        return featuresNormalized
  
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
  
    def train(self, features, labels, learningRate, numIterations):
        num_samples, num_features = features.shape
        # Initialize the parameters with zeros
        self.weights = np.zeros(num_features)
        self.bias = 0
        for iteration in range(numIterations):
            # Compute the linear combination of features and weights
            z = np.dot(features, self.weights) + self.bias
            # Apply the sigmoid activation function
            predictions = self.sigmoid(z)
            # Compute the gradient of the cost function
            dw = (1 / num_samples) * np.dot(features.T, (predictions - labels))
            db = (1 / num_samples) * np.sum(predictions - labels)
            # Update the parameters
            self.weights -= learningRate * dw
            self.bias -= learningRate * db
  
    def predict(self, features):
        z = np.dot(features, self.weights) + self.bias               # Calculate the linear part of the prediction
        predictions = self.sigmoid(z)                                # Apply the sigmoid function to obtain the probability
        return predictions