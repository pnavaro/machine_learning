#
# https://github.com/eriklindernoren/ML-From-Scratch
#

import numpy as np
import math
from utils import normalize, polynomial_features

class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return 

    def grad(self, w):
        return 

class LassoRegression():
    """
    Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.alpha = reg_factor
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        n_features=X.shape[1]

        # Initialize weights randomly [-1/N, 1/N]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.alpha * np.linalg.norm(self.w))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.alpha * np.sign(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


