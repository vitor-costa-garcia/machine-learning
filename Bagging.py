import numpy as np
import BaseModel as BaseModel

class BaggingEnsemble(BaseModel):
	def __init__(self, model: BaseModel, n_models: int, params: dict):
		self.model = model
		self.n_models = n_models
		self.model_params = params
		self.models = []

	def fit(X: np.ndarray, y: np.ndarray):
		pass

	def predict(X: np.array):
		pass
