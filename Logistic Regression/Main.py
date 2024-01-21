# This program is an example of using Logistic Regression for binary classification tasks.
#
# Logistic regression is a popular machine learning algorithm used to classify data into two classes.
# It is commonly used for tasks such as spam detection, disease diagnosis, and sentiment analysis.
# This program implements the Logistic Regression algorithm using the 'Singleton design pattern'.
# The Singleton design pattern ensures that only one instance of the LogisticRegression class is created,
# allowing global access to the instance throughout the program.




import numpy as np
from LogisticRegression import LogisticRegression


def get_user_input():
    num_features = int(input("---> Enter the number of features: "))
    num_samples = int(input("---> Enter the number of training samples: "))
    features = np.zeros((num_samples, num_features))
    labels = np.zeros(num_samples)
    print("*********************************************************************")
    print("---> Enter the features:")
    for i in range(num_samples):
        features[i] = input().split()
    print("*********************************************************************")
    print("---> Enter the labels:")
    for i in range(num_samples):
        labels[i] = int(input())
  
    return features, labels
  

def banner():
    print( """

**********************************************************************************************************
*                                       Welcome to Logistic Regression!                                  *
*                                                                                                        *
*    This program implements the Logistic Regression algorithm using the 'Singleton design pattern'.     *
*   Logistic regression is a popular machine learning algorithm used for binary classification tasks.    *
*      The Singleton design pattern ensures that only one instance of the LogisticRegression class       *
*            is created, allowing global access to the instance throughout the program.                  *
*                                                                                                        *
*           Follow the instructions below to use this program:                                           *
*                                                                                                        *
*                   1. Enter the number of features and training samples.                                *
*                   2. Enter the values for features and labels.                                         *
*                   3. The program will normalize the features.                                          *
*                   4. The logistic regression model will be trained.                                    *
*                   5. The weights and bias of the model will be printed.                                *
*                                                                                                        *
*                   Example:                                                                             *
*                           ---> Enter the number of features: 2                                         *
*                           ---> Enter the number of training samples: 4                                 *
*                           ---> Enter the features:                                                     *
*                                                   1 0                                                  *
*                                                   2 3                                                  *
*                                                   4 1                                                  *
*                                                   3 2                                                  *
*                               ---> Enter the labels:                                                   *
*                                                   0                                                    *
*                                                   1                                                    *
*                                                   0                                                    *
*                                                   1                                                    *
*                                                                                                        *
*           Press Enter after providing the values.                                                      *
**********************************************************************************************************
    """)

def main():
    banner()
    features, labels = get_user_input()
    # Normalize the features
    features_normalized = LogisticRegression.normalizeFeatures(features)
    # Set the learning rate and number of iterations
    learning_rate = 0.01
    num_iterations = 1000
    # Get the instance of the logistic regression model
    model = LogisticRegression.get_instance()
    # Train the logistic regression model
    model.train(features_normalized, labels, learning_rate, num_iterations)
    print("\n*********************************************************************")
    # Print the learned parameters
    print("         ---> Weights:", model.weights)
    print("         ---> Bias:", model.bias)
    print("*********************************************************************\n")
  
  
if __name__ == '__main__':
    main()