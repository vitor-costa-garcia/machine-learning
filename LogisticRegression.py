from BaseModel import BaseModel
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(BaseModel):
	"""Logistic regression is fitted using gradient descent (first order) method or Newton's method (second order)"""
	def __init__(self, lr: float = 0.01, batch_size: int = 20, fit_method: str = 'minigd', tol: float = 10e-6, max_iter: int = 500):
		assert fit_method in ['minigd', 'newton'], "Invalid fit method, must be 'minigd' or 'newton'"
		self.fit_method = fit_method
		self.batch_size = batch_size
		self.tol = tol
		self.max_iter = max_iter
		self.lr = lr

	def _mini_batch_gd(self):
	    sampled_indexes = np.random.choice(self.obs, self.batch_size, replace=False)
	    X_batch = self.X[sampled_indexes]
	    X_bias = np.c_[np.ones(X_batch.shape[0]), X_batch]
	    y_batch = self.y[sampled_indexes].flatten() # Ensure 1D

	    y_hat = self.predict(X_batch)

	    gradient = (1 / self.batch_size) * np.dot(X_bias.T, (y_hat - y_batch))

	    self.coef -= self.lr * gradient

	def _newton_method(self):
		pass

	def fit(self, X: np.array, y: np.array):
		self.X = X
		self.y = y

		self.obs, self.feat = X.shape

		#Starting coefficients as 1s
		self.coef = np.ones(self.feat + 1)

		iter_count = 0
		match self.fit_method:
			case 'minigd':
				while iter_count < self.max_iter:
					old_coef = self.coef.copy()
					self._mini_batch_gd()

					if not (abs(old_coef - self.coef) > self.tol).sum():
						print("Converged to tolerance")
						return

					iter_count += 1
				print(f"Max iterations reached ({self.max_iter}). No convergence to tol={self.tol}")

			case 'newton':
				while iter_count < self.max_iter:
					old_coef = self.coef.copy()
					self._newton_method()

					if not (abs(old_coef - self.coef) > self.tol).sum():
						print("Converged to tolerance")
						return

					iter_count += 1
				print(f"Max iterations reached ({self.max_iter}). No convergence to tol={self.tol}")

	def predict(self, X: np.array):
	    rows = X.shape[0]
	    X_bias = np.c_[np.ones(rows), X]
	    coef_x = np.dot(X_bias, self.coef.T)

	    return np.exp(coef_x) / (1+np.exp(coef_x))

if __name__ == "__main__":
	X = np.concatenate([
		np.random.normal(3, 2, 500),
		np.random.normal(14, 5, 500),
	])

	y = np.concatenate([
		np.zeros(500), np.ones(500),
	])

	plt.scatter(X, y)

	lgr = LogisticRegression(lr=0.01, max_iter=50000, tol=10e-7)
	lgr.fit(X.reshape(-1, 1), y.reshape(-1 ,1))

	sigmoid_x = np.linspace(-5, 20, 100)
	sigmoid_y = lgr.predict(sigmoid_x)
	plt.plot(sigmoid_x, sigmoid_y, color='orange')
	plt.show()