from BaseModel import BaseModel
import numpy as np
import matplotlib.pyplot as plt

class LinearDiscriminantAnalysis(BaseModel):
	def __init__(self):
		# LDA doesnt have any explicit hyperparameters
		pass

	def _multivariate_gaussian_pdf(self, X: np.array):
		likelihood = list()

		# For each class, we compute the likelihood for a given observation X
		# The likelihood is given by the Gaussian pdf
		# In the linear case, the pooled covariance matrix is used for all classes 
		for c in range(self.n_classes):
			X_class_mu = self.means[:, c]
			obs_centralized = X - X_class_mu

			exponent = -0.5 * (obs_centralized @ self.cov_mat_inv @ obs_centralized.T)
			denom = np.sqrt(((2*np.pi)**self.features) * self.cov_mat_det)

			likelihood.append(np.exp(exponent)/denom)

		return np.array(likelihood)

	def _joint_prob(self, X: np.array):
		# The joint probability is given by the product of the prior prob and the likelihood
		# Independent observations P(A, B) = P(B)*P(A|B) A->Observation B->Class label
		likelihood = self._multivariate_gaussian_pdf(X)
		return self.prior_prob * likelihood

	def fit(self, X: np.array, y: np.array):
		self.X, self.y = X, y
		self.observations, self.features = X.shape

		# Count and priors of each class
		unique_values, self.nc = np.unique(self.y, return_counts=True)
		self.n_classes = len(unique_values)
		self.prior_prob = self.nc / len(y)

		# Compute the mean for each feature, for each class
		self.means = np.column_stack([np.mean(self.X[self.y == c], axis = 0) for c in range(self.n_classes)])

		# Pooled covariance matrix (Almost a mean covariance matrix p/class)
		class_cov_matrices = []

		# Calculate each class covariance matrix then sum all and divide by n_observations
		for c in range(self.n_classes):
			X_class = self.X[self.y == c]
			X_class_mu = self.means[:, c]
			X_class_centralized = X_class - X_class_mu
			class_cov_matrices.append((self.nc[c]/self.observations) * np.dot(X_class_centralized.T, X_class_centralized) / (self.nc[c] - 1))

		# Pre computing inverse and determinant
		self.cov_matrix = np.sum(class_cov_matrices, axis=0)
		self.cov_mat_inv = np.linalg.inv(self.cov_matrix)
		self.cov_mat_det = np.linalg.det(self.cov_matrix)
		
	def predict(self, X: np.array):
		# Returns the class which has the max joint probability
		predicted = np.array([np.argmax(self._joint_prob(xn)) for xn in X])
		return predicted
		

if __name__ == "__main__":
	# Generating 3 clusters
	X = np.concatenate([
		np.random.normal(4,1, (125, 2)),
		np.random.normal(-2,1, (125, 2)),
		np.random.normal(-6,2, (250, 2)),
	])

	y = np.concatenate([
		np.ones(125),
		np.zeros(125),
		np.ones(250)*2,
	])

	# LDA
	lda = LinearDiscriminantAnalysis()
	lda.fit(X, y)

	# Contour
	x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
	y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

	xx, yy = np.meshgrid(np.arange(x_min-2, x_max+2, 0.1), np.arange(y_min-2, y_max+2, 0.1))

	grid_points = np.c_[xx.ravel(), yy.ravel()]

	predicted_grid = lda.predict(grid_points)
	Z = predicted_grid.reshape(xx.shape)

	# Plotting
	fig, ax = plt.subplots(1,1)
	ax.contourf(xx, yy, Z, alpha=0.3)
	ax.scatter(X[:, 0], X[:, 1], c=y)
	plt.show()