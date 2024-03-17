import numpy as np


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_prior_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_prior_probs[c] = (len(X_c) + self.alpha) / (len(X) + self.alpha * len(self.classes))
            self.feature_probs[c] = (np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c) + self.alpha * X.shape[1])

    def _predict_single(self, x):

        posteriors = {c: np.log(self.class_prior_probs[c]) + np.sum(np.log(self.feature_probs[c]) * x) for c in self.classes}
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return predictions


# +
from sklearn.datasets import make_classification

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

X -= X.min()
# +
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");


# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# +
from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[6]])

print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])

# +
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
# -

mymodel = NaiveBayes()

mymodel.fit(X_train, y_train)

mymodel.class_prior_probs

# +
y_pred = mymodel.predict(X_test)



# +
y_pred = mymodel.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
# -


