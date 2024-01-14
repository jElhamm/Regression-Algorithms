# This script is the main driver program for using the decision tree regression model.
# It prompts the user to enter input data in the form of X and y values, 
# as well as the maximum depth for the decision tree.



import warnings
import numpy as np
from DecisionTreeRegressorBuilder import DecisionTreeRegressorBuilder


def get_user_input():
    X = input("\n    ---> Enter the values of X (separated by spaces): ")
    X = np.array([float(x) for x in X.split()])                                     # Convert X values to float array
    X = X.reshape(-1, 1)                                                            # Reshape X to have two dimensions
    y = input("    ---> Enter the values of y (separated by spaces): ")
    y = np.array([float(y_val) for y_val in y.split()])                             # Convert y values to float array
    test_X = input("    ---> Enter the values of test_X (separated by spaces): ")
    test_X = np.array([float(x) for x in test_X.split()])                           # Convert test_X values to float array
    test_X = test_X.reshape(-1, 1)                                                  # Reshape test_X to have two dimensions
    return X, y, test_X

def banner():
    print("""
          
    ***************************************************************************************************
    *            (:           Welcome to Decision Tree Regression!             :)                     *
    *                                                                                                 *
    *           This program trains a decision tree regression model using the input data             *
    *        and predicts the values for test data. Follow the instructions below to use it.          *
    *                                                                                                 *
    *               Instructions:                                                                     *
    *                           - Enter the values of X separated by spaces.                          *
    *                           - Enter the values of y separated by spaces.                          *
    *                                                                                                 *
    *               Example:                                                                          *
    *                           X: 1 2 3 4 5                                                          *
    *                           y: 1.5 2.5 3.5 4.5 5.5                                                *
    *                           test_X: 6 7 8 9 10                                                    *
    *                           Enter the maximum depth for the decision tree: 3                      *
    *                                                                                                 *
    ***************************************************************************************************
    """)

def main():
    banner()
    X, y, test_X = get_user_input()
    max_depth = int(input("    ---> Enter the maximum depth for the decision tree: "))
    model = (
        DecisionTreeRegressorBuilder()
        .set_X(X)
        .set_y(y)
        .set_max_depth(max_depth)
        .build()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train()
        predictions = model.predict(test_X)
    print("    ---> Predictions:", np.nan_to_num(predictions, nan=0.0), "\n")
    print("    ***************************************************************************************************\n")


if __name__ == "__main__":
    main()         