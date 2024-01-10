# This program is implementing a Linear Regression Model. 
# It trains the model using input data and predicts values for test data. 
# The program follows the 'Factory Design Pattern' to create the regression model object.



import numpy as np

class LinearRegressionModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.weights = None

    def train(self):
        X_with_bias = np.c_[self.X, np.ones(len(self.X))]                                                   # Add bias column to X
        self.weights = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(self.y)         # Calculate weights

    def predict(self, X):
        X_with_bias = np.c_[X, np.ones(len(X))]                                                             # Add bias column to X
        return X_with_bias.dot(self.weights)                                                                # Predict using weights

class RegressionModelFactory:
    @staticmethod
    def create_model(X, y):
        return LinearRegressionModel(X, y)


def get_user_input():
    X = input("\n    ---> Enter the values of X (separated by spaces): ")
    X = np.array([float(x) for x in X.split()])                                                             # Convert X values to float array
    y = input("    ---> Enter the values of y (separated by spaces): ")
    y = np.array([float(y_val) for y_val in y.split()])                                                     # Convert y values to float array
    test_X = input("    ---> Enter the values of test_X (separated by spaces): ")
    test_X = np.array([float(x) for x in test_X.split()])                                                   # Convert test_X values to float array
    return X, y, test_X

def banner():
    print( """
    *****************************************************************************************
    *                         Welcome to Linear Regression Model!                           *
    *                                                                                       *
    *        This program trains a linear regression model using the input data and         *
    *      predicts the values for test data. Follow the instructions below to use it.      *
    *                                                                                       *
    *        Instructions:                                                                  *
    *        - Enter the values of X separated by spaces.                                   *
    *        - Enter the values of y separated by spaces.                                   *
    *                                                                                       *
    *        Example:                                                                       *
    *        X: 1 2 3 4 5                                                                   *
    *        y: 10 20 30 40 50                                                              *
    *                                                                                       *
    *       Press Enter after providing the values.                                         *
    *****************************************************************************************
    """)

def main():
    banner()
    X, y, test_X = get_user_input()
    model = RegressionModelFactory.create_model(X, y)
    model.train()
    predictions = model.predict(test_X)
    print("    ---> Predictions:", predictions, "\n")
    print("    *****************************************************************************************\n")


if __name__ == "__main__":
    main()