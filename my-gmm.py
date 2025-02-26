import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, n_components, max_iters=100, tol=1e-6):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
    
    def initialize_params(self, X):
        n_samples, n_features = X.shape
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components
        self.resp = np.zeros((n_samples, self.n_components))
    
    def gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]
        diff = X - mean
        return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)) / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
    
    def e_step(self, X):
        for i in range(self.n_components):
            self.resp[:, i] = self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i])
        self.resp /= self.resp.sum(axis=1, keepdims=True)
    
    def m_step(self, X):
        Nk = self.resp.sum(axis=0)
        self.weights = Nk / X.shape[0]
        self.means = (self.resp.T @ X) / Nk[:, np.newaxis]
        for i in range(self.n_components):
            diff = X - self.means[i]
            self.covariances[i] = (self.resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]
    
    def fit(self, X):
        self.initialize_params(X)
        for _ in range(self.max_iters):
            prev_means = self.means.copy()
            self.e_step(X)
            self.m_step(X)
            if np.linalg.norm(self.means - prev_means) < self.tol:
                break
    
    def predict(self, X):
        likelihoods = np.array([self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i]) for i in range(self.n_components)]).T
        return np.argmax(likelihoods, axis=1)
    
    @staticmethod
    def generate_samples(n_samples=500, n_components=3, random_state=42):
        np.random.seed(random_state)
        means = np.random.rand(n_components, 2) * 10
        covariances = [np.eye(2) * np.random.rand() for _ in range(n_components)]
        samples = []
        labels = []
        for i in range(n_components):
            samples.append(np.random.multivariate_normal(means[i], covariances[i], size=n_samples // n_components))
            labels.extend([i] * (n_samples // n_components))
        return np.vstack(samples), np.array(labels)

# Завантаження даних
data = pd.read_csv("sensor_data_phone1.csv")
numeric_data = data[["X", "Y", "Z"]].dropna().values

# Навчання кастомного GMM
n_components = 3
gmm = GMM(n_components=n_components)
gmm.fit(numeric_data)
clusters = gmm.predict(numeric_data)

data["Cluster"] = clusters

# Візуалізація кластерів
plt.scatter(numeric_data[:, 0], numeric_data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Збереження результатів
data.to_csv("sensor_data_with_clusters_MY_GMM.csv", index=False)

# Генерація даних і візуалізація
samples, labels = GMM.generate_samples()
plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap='coolwarm', alpha=0.5)
plt.xlabel("Generated X")
plt.ylabel("Generated Y")
plt.title("Generated Data from Custom GMM")
plt.show()
