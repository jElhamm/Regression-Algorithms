# The program prompts the user to enter values for X, y, alpha, and test_X.
# It then creates an instance of RidgeRegressionModel using the RegressionModelFactory and trains the model with the provided data.
# Finally, it uses the trained model to predict values for the test_X data and prints the predictions.



import numpy as np
from RegressionModelFactory import RegressionModelFactory


def get_user_input():
    X = input("\n    ---> Enter the values of X (separated by spaces): ")
    X = np.array([float(x) for x in X.split()])
    y = input("    ---> Enter the values of y (separated by spaces): ")
    y = np.array([float(y_val) for y_val in y.split()])
    alpha = float(input("    ---> Enter the value of alpha: "))
    test_X = input("    ---> Enter the values of test_X (separated by spaces): ")
    test_X = np.array([float(x) for x in test_X.split()])
    return X, y, alpha, test_X

def banner():
    print("""
    *****************************************************************************************
    *                         Welcome to Ridge Regression Model!                            *
    *                                                                                       *
    *        This program trains a ridge regression model using the input data and          *
    *           predicts the values for test data. Follow the instructions below            *
    *                                to use it.                                             *
    *                                                                                       *
    *        Instructions:                                                                  *
    *        - Enter the values of X separated by spaces.                                   *
    *        - Enter the values of y separated by spaces.                                   *
    *        - Enter the value of alpha, a hyperparameter for controlling regularization    *
    *          strength.                                                                    *
    *                                                                                       *
    *        Example:                                                                       *
    *        X: 1 2 3 4 5                                                                   *
    *        y: 10 20 30 40 50                                                              *
    *        Alpha : 0.1                                                                    *
    *        test_X: 8 5 6                                                                  *
    *                                                                                       *
    *       Press Enter after providing the values.                                         *
    *****************************************************************************************
    """)

def main():
    banner()
    X, y, alpha, test_X = get_user_input()
    model = RegressionModelFactory.create_model(X, y, alpha)
    model.train()
    predictions = model.predict(test_X)
    print("    ---> Predictions:", predictions, "\n")
    print("    *****************************************************************************************\n")


if __name__ == "__main__":
    main()