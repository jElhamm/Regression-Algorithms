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
 