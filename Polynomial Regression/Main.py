# This code is a command-line program that utilizes the Polynomial Regression Model 
# to perform regression on user-provided data.



import numpy as np
from RegressionModelFactory import RegressionModelFactory


def get_user_input():
    X = input("\n    ---> Enter the values of X (separated by spaces): ")
    X = np.array([float(x) for x in X.split()])                                     # Convert X values to float array
    y = input("    ---> Enter the values of y (separated by spaces): ")
    y = np.array([float(y_val) for y_val in y.split()])                             # Convert y values to float array
    test_X = input("    ---> Enter the values of test_X (separated by spaces): ")
    test_X = np.array([float(x) for x in test_X.split()])                           # Convert test_X values to float array
    degree = int(input("    ---> Enter the degree of polynomial regression: "))     # Get degree of polynomial regression
    return X, y, test_X, degree

def banner():
    print("""
    *****************************************************************************************
    *                      Welcome to Polynomial Regression Model!                          *
    *                                                                                       *
    *    This program trains a polynomial regression model using the input data and         *
    *    predicts the values for test data. Follow the instructions below to use it.        *
    *                                                                                       *
    *    Instructions:                                                                      *
    *    - Enter the values of X separated by spaces.                                       *
    *    - Enter the values of y separated by spaces.                                       *
    *    - Enter the degree of polynomial regression.                                       *
    *                                                                                       *
    *    Example:                                                                           *
    *    ---> Enter the values of X (separated by spaces): 1 2 3 4 5                        *
    *    ---> Enter the values of y (separated by spaces): 2 4 6 8 10                       *
    *    ---> Enter the values of test_X (separated by spaces): 6 7 8                       *
    *    ---> Enter the degree of polynomial regression: 2                                  *
    *                                                                                       *
    *****************************************************************************************
    """)

def main():
    banner()
    X, y, test_X, degree = get_user_input()
    model_factory = RegressionModelFactory()
    model = model_factory.create_model(X, y, degree)
    model.train()
    predictions = model.predict(test_X)
    print("    ---> Predictions:", predictions, "\n")
    print("    *****************************************************************************************\n")


if __name__ == "__main__":
    main()