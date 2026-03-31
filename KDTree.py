import numpy as np
from BaseModel import BaseModel

class KDTreeNode:
	def __init__(self, active_samples: np.array):
		self.active_samples = active_samples

		self.split_feature = None
		self.split_value = None

		self.left = None
		self.right = None

class KDTree(BaseModel):
	def __init__(self, min_samples_leaf: int):
		self.min_samples_leaf = min_samples_leaf

	def _sup_fit(self, node: KDTreeNode, feature: int = 0):
		active_indices = np.where(node.active_samples)[0]
		sorted_active = active_indices[np.argsort(self.X[active_indices, feature])]

		median_split = len(sorted_active) // 2
		median_idx = sorted_active[median_split]

		left_i = sorted_active[:median_split]
		right_i = sorted_active[median_split:]

		if len(left_i) < self.min_samples_leaf or len(right_i) < self.min_samples_leaf:
		    return node

		node.split_feature = feature
		node.split_value = self.X[median_idx, feature]
		node.point = self.X[median_idx]

		left_active_samples = np.zeros_like(node.active_samples)
		right_active_samples = np.zeros_like(node.active_samples)

		left_active_samples[left_i] = True
		right_active_samples[right_i] = True

		left_active_samples &= node.active_samples
		right_active_samples &= node.active_samples

		node_l = KDTreeNode(left_active_samples)
		node_r = KDTreeNode(right_active_samples)

		next_feature = (feature+1) % self.features

		node.left = self._sup_fit(node_l, next_feature)
		node.right = self._sup_fit(node_r, next_feature)

		return node

	def fit(self, X: np.array, y: np.array = None):
		assert X.shape[0] > (self.min_samples_leaf*2), f"Não é possível fazer nenhuma divisão com esse conjunto de dados. min_samples_leaf: {self.min_samples_leaf}"

		self.X = X
		self.observations, self.features = X.shape
		self.sorted_features = np.argsort(X, axis=0)

		self.root = KDTreeNode(np.ones(self.observations, dtype=bool))

		self._sup_fit(self.root)

	def _inorder(self, node:KDTreeNode):
		if node == None:
			return

		self._inorder(node.left)

		if node.left == None and node.right == None:
			print(f"Leaf node: {node.active_samples.sum()} samples")
		else:
			print(f"Node: feature->{node.split_feature} split_val->{node.split_value}")

		self._inorder(node.right)

	def _find_k_sup(self, X: np.array, node: KDTreeNode):
		if node.left == None and node.right == None:
			#reached the leaf

		split_feat = node.split_feature
		split_val = node.split_value

		if X[node.split_feature] > split_val:
			pass

		else:
			pass 

	def find_k_neighbors(self, k: int, X: np.ndarray):
		self.k_closest = []
		self._find_k_sup(self.root)


if __name__ == "__main__":
	X = np.random.normal(0, 3, (300, 3))
	kdt = KDTree(1)
	kdt.fit(X)

	kdt._inorder(kdt.root)



