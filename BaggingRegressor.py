import numpy as np
from BaseModel import BaseModel
from Bootstrapping import Bootstrap
from DecisionTreeRegressor import DecisionTreeRegressor

class BaggingRegressor(BaseModel):
	def __init__(self, model: BaseModel, n_models: int, bootstrap_frac: float, params: dict):
		self.model = model
		self.n_models = n_models
		self.bootstrap_frac = bootstrap_frac
		self.model_params = params

	def fit(self, X: np.ndarray, y: np.ndarray):
		self.models = []

		bootstrap = Bootstrap(self.n_models, self.bootstrap_frac)
		bootstrap_splits = bootstrap.fit_transform(X)
		for i_model in range(self.n_models):
			bootstrap_X, bootstrap_y = X[bootstrap_splits[i_model]], y[bootstrap_splits[i_model]]
			new_model = self.model(**self.model_params)
			new_model.fit(bootstrap_X, bootstrap_y)
			self.models.append(new_model)

	def predict(self, X: np.array):
		predictions = np.array([model.predict(X) for model in self.models])
		return np.mean(predictions, axis=0)

if __name__ == "__main__":
	X = np.random.normal(0, 1, (3000, 2))
	y = np.random.normal(2, 1, 3000)

	params = {
	    'bins': 255,
	    'max_n_leaves': 63,
	    'min_samples_node': 10,
	    'max_depth': 7
	}

	X_test = np.random.normal(0, 1, (400, 2))
	y_test = np.random.normal(2, 1, 400)

	bgreg = BaggingRegressor(model = DecisionTreeRegressor, n_models = 10, bootstrap_frac = 0.6, params = params)

	bgreg.fit(X, y)
	predictions = bgreg.predict(X_test)

	print(predictions)
