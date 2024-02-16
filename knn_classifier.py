# +
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs


# -

class KNNClassifier:

	def __init__(self, k=5):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def euclidean_distance(self, x1, x2):
		return np.sqrt(np.sum(x1 - x2) ** 2)

	def predict(self, X):
		y_pred = [self._predict_single(x) for x in X]
		return np.array(y_pred)

	def _predict_single(self, x):
		# Calculate distances between x and all examples in the training set
		distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

		# Get the indices of the k-nearest neighbors
		k_indices = np.argsort(distances)[:self.k]

		# Get the labels of the k-nearest neighbors
		k_nearest_labels = [self.y_train[i] for i in k_indices]

		# Return the most common class label among the k-nearest neighbors
		most_common = np.bincount(k_nearest_labels).argmax()

		return most_common


# Generate sample data
n_samples = 400
n_components = 3

X, y_true = make_blobs(
    n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0
)
X = X[:, ::-1]

# Plot init seeds along side sample data
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]


for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

model = KNNClassifier(5)
model.fit(X, y_true)

y_pred = model.predict(X)

for k, col in enumerate(colors):
    cluster_data = y_pred == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
plt.xticks([])
plt.yticks([])
