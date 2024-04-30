import numpy as np
import scipy as sp

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

    def pvalues(self, x, y):
        sse = np.sum((self.predict(x) - y) ** 2, axis=0) / float(x.shape[0] - x.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        t = self.w / se
        self.p = 2 * (1 - sp.stats.t.cdf(np.abs(t), y.shape[0] - x.shape[1]))



