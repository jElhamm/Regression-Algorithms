# It prompts the user to input data for training and testing the SVR model.
# The user provides input values for X (features), y (target values), kernel type, regularization parameter (C), and a margin of tolerance (epsilon).
# The script uses the input data to train the SVR model and then predicts the values for the test data.



import numpy as np
from RegressionModelFactory import RegressionModelFactory


def get_user_input():
    X = input("\n    ---> Enter the values of X (separated by spaces): ")
    X = np.array([float(x) for x in X.split()])
    y = input("    ---> Enter the values of y (separated by spaces): ")
    y = np.array([float(y_val) for y_val in y.split()])
    kernel = input("    ---> Enter the kernel type (default='rbf'): ") or 'rbf'
    C = float(input("    ---> Enter the value of C (default=1.0): ") or 1.0)
    epsilon = float(input("    ---> Enter the value of epsilon (default=0.1): ") or 0.1)
    test_X = input("    ---> Enter the values of test_X (separated by spaces): ")
    test_X = np.array([float(x) for x in test_X.split()])
    return X, y, kernel, C, epsilon, test_X

def banner():
    print("""
          
    *****************************************************************************************
    *                         Welcome to Support Vector Regression Model!                   *
    *                                                                                       *
    *      This program trains a Support Vector Regression model using the input data       *
    *         and predicts the values for test data. Follow the instructions below          *
    *                                to use it.                                             *
    *                                                                                       *
    *        Instructions:                                                                  *
    *        - Enter the values of X separated by spaces.                                   *
    *        - Enter the values of y separated by spaces.                                   *
    *        - Enter the kernel type (default='rbf').                                       *
    *        - Enter the value of C, a regularization parameter (default=1.0).              *
    *        - Enter the value of epsilon, a margin of tolerance (default=0.1).             *
    *                                                                                       *
    *        Example:                                                                       *
    *        X: 1 2 3 4 5                                                                   *
    *        y: 10 20 30 40 50                                                              *
    *        Kernel: rbf                                                                    *
    *        C: 1.0                                                                         *
    *        Epsilon: 0.1                                                                   *
    *        test_X: 8 7 9                                                                  *
    *                                                                                       *
    *       Press Enter after providing the values.                                         *
    *****************************************************************************************
    """)

def main():
    banner()
    X, y, kernel, C, epsilon, test_X = get_user_input()
    model = RegressionModelFactory.create_model(X, y, kernel, C, epsilon)
    model.fit()
    predictions = model.predict(test_X)
    print("    ---> Predictions:", predictions, "\n")
    print("    *****************************************************************************************\n")

if __name__ == "__main__":
    main()