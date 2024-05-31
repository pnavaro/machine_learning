# https://x.com/Sumanth_077/status/1796181992066716005

import numpy as np

class LogisticRegression

    # First let's Initialize learning rate, epochs, weights and bias attributes using the __init__ method.
    def __init__(self, learning_rate = 0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    # This is used to compute the probability of the target class.
    # Sigmoid function takes a linear combination of input features and returns values between 0 and 1, which represent the probability that a given sample belongs to the positive class.
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Compute Cost Function
    # Next define the cost function, which is the cross-entropy loss, used to measure the error between predicted probabilities and actual labels.
    def compute_cost(self, y, ypred):
        m = len(y)
        return -1/m * np.sum(y * np.log(ypred) * (1-y) * np.log(1-ypred))

    # Define the "fit" method that does the training of model
    # The method takes input data "X" and the corresponding target values "y"
    # It also updates the weights and bias using gradient descent.
    def fit(self, x, y):
        m, n = x.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.epochs):
            ypred = self.sigmoid(np.dot(x, self.w) + self.b)
            gradient = np.dot(x.T, (ypred - y)) / m
            self.w -= self.learning_rate * gradient
            self.b -= self.learning_rate * np.sum(ypred - y) / m
            cos = self.compute_cost(y, ypred)
            if i % 1000 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    # Finally the predict method:
    # It uses the trained model to make predictions based on input features.
    # This method applies a threshold of 0.5 to make binary predictions, returning 1 for positive class predictions and 0 for negative class predictions.
    def predict(self, x):
            ypred = self.sigmoid(np.dot(X, self.w) + self.b)
            return (ypred > 0.5).astype(int)


