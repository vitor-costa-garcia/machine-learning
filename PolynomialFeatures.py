from BaseModel import BaseModel
import numpy as np

class PolynomialFeatures(BaseModel):
	def __init__(self, degrees: int):
		self.degrees = degrees

	def fit_transform(X: np.array):
		rows, cols = X.shape
		new_X = np.ones((rows, cols*(degrees + 1)))

		for col in range(col):
			pass


if __name__ == "__main__":
	pass