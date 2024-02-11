import numpy as np

class LogisticRegression
    def __init__(self, learning_rate = 0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, y, ypred):
        m = len(y)
        return -1/m * np.sum(y * np.log(ypred) * (1-y) * np.log(1-ypred))

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

        def predict(self, x):
            ypred = self.sigmoid(np.dot(X, self.w) + self.b)
            return (ypred > 0.5).astype(int)


