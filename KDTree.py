import numpy as np
from BaseModel import BaseModel

class KDTreeNode:
	def __init__(self, active_samples: np.array):
		self.split_feature = split_feature
		self.split_value = split_value

		self.left = None
		self.right = None

		self.active_samples = active_samples 

class KDTree(BaseModel):
	def __init__(self, min_samples_leaf: int):
		self.min_samples_leaf = min_samples_leaf

	def _sup_fit(node: KDTreeNode, feature: int):
		median_split = self.sorted_features[:, feature]

	def fit(self, X: np.array, y: np.array = None):
		self.observations, self.features = X.shape
		self.sorted_features = np.argsort(X, axis=0)

		self.root = KDTreeNode(np.ones(self.observations))

		if self.observations//2 < self.min_samples_node:
			return

		self._sup_fit(self.root)

if __name__ == "__main__":
	X = np.random.normal(0, 3, (30, 3))
	kdt = KDTree(5)
	kdt.fit(X)



