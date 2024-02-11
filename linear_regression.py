import numpy as np

class LinearRegression

    def __init__(self, learning_rate = 0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def fit(self, x, y):
        m, n = x.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):
            ypred = self.predic(x)
            dw = np.dot(x.T, (ypred - y)) / m
            db = np.sum(ypred - y) / m
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
