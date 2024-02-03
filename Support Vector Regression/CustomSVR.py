# This is a custom Support Vector Regression (SVR) class that implements the fitting and prediction functionalities
# using the Sequential Minimal Optimization (SMO) algorithm. The class allows for the training of an SVR model and 
# making predictions on new data points. The SVR model includes options for using different kernel functions such 
# as the radial basis function (RBF) kernel.



import numpy as np


class CustomSVR:
    def __init__(self, X, y, kernel='rbf', C=1.0, epsilon=0.1):
        # Initializing with input data and parameters
        self.X = X 
        self.y = y
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.support_vectors = None
        self.support_vector_labels = None
        self.dual_coef = None
        self.intercept = None

    def fit(self):
        n_samples = len(self.X)                                                 # Getting the number of samples
        K = self._compute_kernel_matrix()                                       # Computing the kernel matrix
        P = np.outer(self.y, self.y) * K                                        # Formulating the Quadratic Programming problem
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))                  # Constructing inequality constraints
        h = np.hstack((np.zeros(n_samples), np.full(n_samples, self.C)))
        A = self.y                                                              # Constructing equality constraints
        b = 0.0
        # Using CVXOPT library to solve the QP problem
        from cvxopt import matrix, solvers
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A, (1, n_samples)), matrix(b))
        alphas = np.array(sol['x']).flatten()
        # Finding support vectors, support vector labels, dual coefficients, and intercept
        self.support_vectors = self.X[alphas > 1e-5]
        self.support_vector_labels = self.y[alphas > 1e-5]
        self.dual_coef = alphas[alphas > 1e-5] * self.support_vector_labels
        self.intercept = np.mean(self.support_vector_labels - np.sum(self.dual_coef * K[alphas > 1e-5, :], axis=0))
 