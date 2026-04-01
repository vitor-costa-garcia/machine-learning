from BaseModel import BaseModel
import numpy as np
import matplotlib.pyplot as plt

class QuadraticDiscriminantAnalysis(BaseModel):
	def __init__(self):
		# QDA doesnt have any explicit hyperparameters
		pass

	def _calculate_cov_mat_invdet(self):
		# Stores the inverse and determinant of each class covariance matrix
		# Since these operations are expensive, we calculate them only once
		self.covmat_inv = list()
		self.covmat_det = list()

		for c in range(self.n_classes):
			cov_mat_c = self.cov_matrices[c]
			self.covmat_inv.append(np.linalg.inv(cov_mat_c))
			self.covmat_det.append(np.linalg.det(cov_mat_c))

	def _multivariate_gaussian_pdf(self, X: np.array):
		likelihood = list()

		# For each class, we compute the likelihood for a given observation X
		# The likelihood is given by the Gaussian pdf
		# In the quadratic case, we compute a covariance matrix for each class 
		for c in range(self.n_classes):
			X_class_mu = self.means[:, c]
			obs_centralized = X - X_class_mu

			exponent = -0.5 * (obs_centralized @ self.covmat_inv[c] @ obs_centralized.T)
			denom = np.sqrt(((2*np.pi)**self.features) * self.covmat_det[c])

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

		# Covariance matrices for each class
		self.cov_matrices = []

		# Calculate each class covariance matrix then store inside self.cov_matrices
		for c in range(self.n_classes):
			X_class = self.X[self.y == c]
			X_class_mu = self.means[:, c]
			X_class_centralized = X_class - X_class_mu
			self.cov_matrices.append(np.dot(X_class_centralized.T, X_class_centralized) / (self.nc[c] - 1))

		# Pre-compute the inverse and determinant of each covariance matrix
		self._calculate_cov_mat_invdet()

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

	# QDA
	qda = QuadraticDiscriminantAnalysis()
	qda.fit(X, y)

	# Contour
	x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
	y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

	xx, yy = np.meshgrid(np.arange(x_min-2, x_max+2, 0.1), np.arange(y_min-2, y_max+2, 0.1))

	grid_points = np.c_[xx.ravel(), yy.ravel()]

	predicted_grid = qda.predict(grid_points)
	Z = predicted_grid.reshape(xx.shape)

	# Plotting
	fig, ax = plt.subplots(1,1)
	ax.contourf(xx, yy, Z, alpha=0.3)
	ax.scatter(X[:, 0], X[:, 1], c=y)
	plt.show()