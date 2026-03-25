from BaseModel import BaseModel
import numpy as np

class Bootstrap(BaseModel):
	def __init__(self, n: int, frac: float):
		self.frac = frac
		self.n = n

	def fit_transform(self, X: np.array):
		samples = X.shape[0]
		bootstrap_samples = int(samples * self.frac)

		bootstrap_splits = np.array([
			np.random.choice(samples, bootstrap_samples, replace=True) for _ in range(self.n)
		])

		return bootstrap_splits