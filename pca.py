import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, data):
        self.mean = np.mean(data, axis = 0)
        normalized_data = data - self.mean
        covariance_matrix = np.cov(normalized_data, rowvar = False)
        eigenvalues, self.eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

    def transform(self, data):
        normalized_data = data - self.mean
        reduced_data = np.dot(normalized_data, self.eigenvectors[:, :self.n_components])
        return reduced_data

    def inverse_transform(self, reduced_data):
        reconstructed_data = np.dot(reduced_data, self.eigenvectors[:, :self.n_components].T) + self.mean
        return reconstructed_data


data = np.random.random(size=(100,3))

pca = PCA(2)

pca.fit(data)

pca.transform(data)


