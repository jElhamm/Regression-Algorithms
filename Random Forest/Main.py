import warnings
import numpy as np
from RandomForestRegressor import RandomForestRegressor
warnings.filterwarnings("ignore")


def get_user_input():
    n_estimators = int(input("    ---> Enter the number of estimators: "))
    max_depth = int(input("    ---> Enter the maximum depth (or leave empty for default): ") or "None")
    n_samples = int(input("    ---> Enter the number of samples (X and y): "))
    X = np.zeros((n_samples, 1))
    y = np.zeros((n_samples,))
    print("\n    ---> Enter values for X:")
    for i in range(n_samples):
        X[i] = float(input(f"        -  X[{i}]: "))
    print("\n    ---> Enter values for y:")
    for i in range(n_samples):
        y[i] = float(input(f"        -  y[{i}]: "))
    return X, y, n_estimators, max_depth


def banner():
    print( """

    **********************************************************************************
    *                   Welcome to Random Forest Regression!                         *
    *                                                                                *
    *      This program trains a random forest regression model using the            *
    *            input data and predicts the values for test data.                   *
    *                                                                                *
    *      Follow the instructions below to use this program:                        *
    *                                                                                *
    *            - Enter the number of estimators                                    *
    *            - Enter the maximum depth                                           *
    *            - Enter the number of samples for X and y                           *
    *            - Enter values for X                                                *
    *            - Enter values for y                                                *
    *                                                                                *
    *      Example:                                                                  *
    *          ---> Enter the number of estimators: 100                              *
    *          ---> Enter the maximum depth (or leave empty for default): 3          *
    *          ---> Enter the number of samples (X and y): 5                         *
    *          ---> Enter values for X:                                              *
    *              -  X[0]: 1                                                        *
    *              -  X[1]: 2                                                        *
    *              -  X[2]: 3                                                        *
    *              -  X[3]: 4                                                        *
    *              -  X[4]: 5                                                        *
    *          ---> Enter values for y:                                              *
    *              -  y[0]: 10                                                       *
    *              -  y[1]: 20                                                       *
    *              -  y[2]: 30                                                       *
    *              -  y[3]: 40                                                       *
    *              -  y[4]: 50                                                       *
    *          ---> Enter test sample values: 3                                      *
    *              -  X_test[0]: 2.5                                                 *
    *              -  X_test[1]: 3.4                                                 *
    *              -  X_test[2]: 4.5                                                 *
    *                                                                                *
    *      Press Enter after providing the values.                                   *
    **********************************************************************************
    """)

def main():
    banner()
    X, y, n_estimators, max_depth = get_user_input()
    # Create and train the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X, y)
    # Make predictions
    n_test_samples = int(input("\n    ---> Enter the number of test samples: "))
    X_test = np.zeros((n_test_samples, 1))
    print("\n    ---> Enter test sample values:")
    for i in range(n_test_samples):
        X_test[i] = float(input(f"        -  X_test[{i}]: "))

    predictions = model.predict(X_test)
    print("\n**********************************************************************************")
    print("    ---> Predictions:", predictions)
    print("**********************************************************************************\n")


if __name__ == '__main__':
    main()