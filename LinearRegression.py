import matplotlib.pyplot as plt
import numpy as np

class LinearRegression():
	"""Implementation using mini-match GD"""
	def __init__(self, max_iter: int, batch_size: int, tol: float, learning_rate: float):
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.tol = tol
		self.lr = learning_rate

	def _mini_batch_gd(X: np.ndarray, y: np.ndarray):
		rows, cols = X.shape
		batch_idx = np.random.choice(rows, size=self.batch_size, replace=False)

		batch = X[batch_idx]

		predicted = self.predict(batch)

		batch[:, cols] = np.ones()
		mean_sum_residuals = (1/self.batch_size) * np.sum(predicted - y)

		batch *= mean_sum_residuals

		print(mean_sum_residuals.shape)


	def _build_params(self, n_coef: int):
		self.coef = np.zeros(n_coef + 1)		

	def fit(self, X: np.ndarray, y: np.ndarray):
		print(X.shape)
		rows, cols = X.shape
		assert rows <= self.batch_size, "O parâmetro batch size deve ser menor ou igual que o número de observações"
		_build_params(cols)

		prev_error = float('inf')

		# while abs(new_error - prev_error) > tol:
		self._mini_batch_gd(X, y)


		pass

	def predict(self, X: np.ndarray):
		rows, cols = X.shape

		#Adding row for bias
		X_w_bias = X.copy()
		X_w_bias[:, cols] = np.ones(rows)
		return np.dot(X, self.coef)

if __name__ == "__main__":
	N_POINTS = 200

	X = np.atleast_2d(np.random.uniform(-200, 200, N_POINTS)).T
	y = X*4 + np.random.normal(0,200, N_POINTS)

	plt.scatter(X.reshape(-1, 1), y)
	plt.show()

	linreg = LinearRegression(100, 10, 10e-4, 0.05)
	linreg.fit(X, y)
