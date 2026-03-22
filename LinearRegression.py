import matplotlib.pyplot as plt
import numpy as np

class LinearRegression():
	"""Implementation using mini-match GD"""
	def __init__(self, max_iter: int, batch_size: int, tol: float, learning_rate: float):
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.tol = tol
		self.lr = learning_rate

	def _mini_batch_gd(self, X: np.ndarray, y: np.ndarray):
	    rows, cols = X.shape
	    batch_idx = np.random.choice(rows, size=self.batch_size, replace=False)

	    X_batch = X[batch_idx]
	    y_batch = y[batch_idx]

	    predictions = self.predict(X_batch)

	    residuals = predictions - y_batch

	    # bias only for gradient
	    X_batch_bias = np.c_[X_batch, np.ones(self.batch_size)]

	    grad = (X_batch_bias.T @ residuals) / self.batch_size
	    print(grad)

	    self.coef -= self.lr * grad

	def _build_params(self, n_coef: int):
		self.coef = np.zeros(n_coef + 1)		

	def fit(self, X: np.ndarray, y: np.ndarray):
		rows, cols = X.shape
		assert rows >= self.batch_size, "O parâmetro batch size deve ser menor ou igual que o número de observações"
		self._build_params(cols)

		X_mean = X.mean(axis=0)   # shape (d,)
		X_std = X.std(axis=0)     # shape (d,)

		y_mean = y.mean(axis=0)   # shape (d,)
		y_std = y.std(axis=0)     # shape (d,)

		X_norm = (X - X_mean) / X_std
		y_norm = (y - y_mean) / y_std

		prev_coef = self.coef.copy()

		for i in range(self.max_iter):
			self._mini_batch_gd(X_norm, y_norm)

		self.coef[:-1] *= y_std
		self.coef[:-1] /= X_std

		self.coef[-1] = y_mean + self.coef[-1]*y_std - np.dot(X_mean, self.coef[:-1])

	def predict(self, X: np.ndarray):
	    rows = X.shape[0]
	    X_bias = np.c_[X, np.ones(rows)]
	    return np.dot(X_bias, self.coef)

if __name__ == "__main__":
	N_POINTS = 200

	#Generating exponential function w/noise
	X = np.random.uniform(-4, 20, N_POINTS)
	y = X**2 + np.random.normal(0,12, N_POINTS)
	X = np.atleast_2d(X).T

	#Linear regression model
	linreg = LinearRegression(100, 30, 10e-4, 0.05)
	linreg.fit(X, y)
	slope, bias = linreg.coef

	#Plotting regression line
	fig, ax = plt.subplots()

	minx, miny, maxx, maxy = min(X), min(X)*slope + bias, max(X), max(X)*slope + bias
	ax.set_title('Linear regression adjusted with mini-batch GD')
	ax.scatter(X.reshape(-1, 1), y)
	ax.plot([minx, maxx], [miny, maxy], 'red')

	plt.show()
