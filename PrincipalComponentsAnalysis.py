import numpy as np
import matplotlib.pyplot as plt

class PrincipalComponentAnalysis():
	def __init__(self, n_components: int):
		# Number of principal components
		self.n_components = n_components

	def fit_transform(self, X: np.ndarray):
		# Centralizing the data
		X_center = X - np.mean(X, axis = 0)

		# Covariance matrix of X_center
		cov_mat = np.cov(X_center, rowvar=False)

		#Eigenvalues and eigenvectors of X_center covariance matrix from highest to lowest
		eigvals, eigvecs = np.linalg.eig(cov_mat)
		idx = eigvals.argsort()[::-1]
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:,idx]

		# n directions with highest variance
		pca_matrix = np.array(eigvecs[:,:self.n_components])

		# Projecting X_center on directions with highest variance
		result = np.dot(X_center, pca_matrix)

		return result

if __name__ == "__main__":
	pca = PrincipalComponentAnalysis(n_components = 2)

	X = np.concat([
		np.random.normal(4, 0.7, (100,3)),
		np.random.normal(-5, 0.7, (100,3)),
		np.random.normal(0, 1.3, (100,3))
	])

	groups = np.concat([
		np.zeros(100),
		np.ones(100),
		np.ones(100)*2,
	])

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d', c=groups)
	ax.scatter(X[:,0], X[:,1], X[:,2])
	plt.show()

	pca_data = pca.fit_transform(X)

	fig, ax = plt.subplots()
	ax.scatter(pca_data[:, 0], pca_data[:, 1], c=groups)
	plt.show()
