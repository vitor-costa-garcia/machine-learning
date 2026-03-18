import numpy as np
from Kmeans import Kmeans

class GaussianMixtureModel():
	def __init__(self, n_components: int):
		self.n_components = n_components

	def fit_transform(self):
		#Using kmeans to initialize gaussians parameters u_k, sigma_k and pi_k
		kmeans = Kmeans(self.n_components)