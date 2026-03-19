import numpy as np
from Kmeans import Kmeans
import matplotlib.pyplot as plt

class GaussianMixtureModel():
	def __init__(self, n_components: int, tol: float):
		self.n_components = n_components
		self.tol = tol

	def _multivariate_gaussian_pdf(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
		d = len(mu)

		det_sig = np.linalg.det(sigma)
		sig_inv = np.linalg.inv(sigma)

		diff = X - mu

		norm_const = 1 / ( (2*np.pi)**(d/2) * np.sqrt(det_sig) )
		exponent = -0.5 * np.sum(np.dot(diff, sig_inv) * diff, axis=1)

		return norm_const * np.exp(exponent)

	def _e_step(self, X: np.ndarray, mixing_coef: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray):
		N = X.shape[0]
		K = self.n_components

		resp = np.ones((N, K))

		for k in range(K):
			resp[:, k] = self._multivariate_gaussian_pdf(X, mu[k], cov_mat[k]) * mixing_coef[k]

		sum_resp_k = np.sum(resp, axis=1)

		return resp/sum_resp_k[:, np.newaxis]

	def _m_step(self, X: np.ndarray, resp: np.ndarray):
		N = X.shape[0]
		K = self.n_components
		N_k = np.sum(resp, axis=0)

		new_mixing_coef = N_k/N

		new_means = np.dot(resp.T, X) / N_k[:, np.newaxis]

		new_cov_mat = list()

		for k in range(K):
			diff = X - new_means[k]

			new_cov = np.dot((resp[:, k] * diff.T), diff) / N_k[k]
			new_cov_mat.append(new_cov)

		new_cov_mat = np.array(new_cov_mat)

		return new_mixing_coef, new_means, new_cov_mat

	def _log_likekihood(self, X: np.ndarray, mixing_coef: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
		N = X.shape[0]
		K = self.n_components

		w_pdfs = np.ones((N, K))

		for k in range(K):
			w_pdfs[:, k] = self._multivariate_gaussian_pdf(X, mu[k], sigma[k]) * mixing_coef[k]

		sum_w_pdfs_k = np.sum(w_pdfs, axis=1)
		log_sum_w_pdfs = np.sum(np.log(sum_w_pdfs_k))

		return log_sum_w_pdfs

	def fit_transform(self, X: np.ndarray):
		#Using kmeans to initialize gaussians parameters u_k, sigma_k and pi_k
		kmeans = Kmeans(self.n_components, 100, 10e-5)
		initial_groups = kmeans.fit_predict(X)

		K = self.n_components
		N = len(initial_groups)

		#Creates a one hot encoding for each observation
		one_hot = np.eye(K)[initial_groups]

		# Vectorized N_k (counts/weights per cluster)
		N_k = np.sum(one_hot, axis=0)

		# Vectorized mixing_coef (Equation 9.22)
		mixing_coef = N_k / N

		# Vectorized mean_k (Equation 9.17 equivalent)
		mean_k = np.dot(one_hot.T, X) / N_k[:, np.newaxis]

		# Covariance matrices (Not vectorized)
		cov_mat_k = list()

		for k in range(self.n_components):
			X_k = X[initial_groups == k]
			cov_mat_k.append(np.cov(X_k, rowvar=False))

		cov_mat_k = np.array(cov_mat_k)

		last_log_l = float("inf")

		while True:
			resp = self._e_step(X, mixing_coef, mean_k, cov_mat_k)
			mixing_coef, mean_k, cov_mat_k = self._m_step(X, resp)

			current_log_l = self._log_likekihood(X, mixing_coef, mean_k, cov_mat_k)
			if abs(current_log_l - last_log_l) <= self.tol:
				break
			last_log_l = current_log_l

		labels = np.argmax(resp, axis=1)

		return labels

if __name__ == "__main__":
	X = np.concatenate([
	    np.random.normal(loc=0, scale=1, size=(100, 3)),
	    np.random.normal(loc=5, scale=1, size=(100, 3)),
	    np.random.normal(loc=-5, scale=1, size=(100, 3)),
	])

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_title("Original data")
	ax.scatter(X[:, 0], X[:, 1], X[:, 2])	
	plt.show()

	#Gaussian mixture model
	gmm = GaussianMixtureModel(n_components=3, tol=10e-3)
	classes = gmm.fit_transform(X)

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_title("Clustering with Gaussian Mixture Model k=3")
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=classes)	
	plt.show()


