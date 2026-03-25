import numpy as np

class DecisionTreeRegressorNode:
	def __init__(self, hist:np.array, active_samples: np.array):
		self.hist = hist
		self.active_samples = active_samples

		self.split_feature = None
		self.split_threshold = None

		self.left = None
		self.right = None

class DecisionTreeRegressor:
	def __init__(self, bins: int, max_n_leaves: int, min_samples_node: int, max_depth: int):
		self.bins = bins
		self.max_n_leaves = max_n_leaves
		self.min_samples_node = min_samples_node
		self.max_depth = max_depth

		self.n_leaves = 1

	def _build_hist(self, active_samples):
		# For a regressor tree, the histogram must store the sum of elements
		hist = list()

		for f_i in range(self.features):
			y_sorted = self.y[self.sorted_features[:, f_i]]
			active_samples_sorted = active_samples[self.sorted_features[:, f_i]]

			feature_bins_sumsize = []
			for i in range(self.bins):
				low, high = self.hist_intervals[i], self.hist_intervals[i + 1]
				y_binned = y_sorted[low:high]
				active_y = active_samples_sorted[low:high]

				sum_bin_active = y_binned[active_y].sum()
				sqrd_sum_bin_active = (y_binned[active_y]**2).sum()
				size_bin_active = active_y.sum()

				feature_bins_sumsize.append(np.array([sum_bin_active, sqrd_sum_bin_active, size_bin_active]))

			hist.append(np.array(feature_bins_sumsize.copy()))

		hist = np.array(hist)
		return hist

	def _find_best_split(self, node: DecisionTreeRegressorNode):
		best_error, split_feature, split_bin = float("inf"), None, None

		if node.active_samples.sum() < 2*self.min_samples_node:
			return None, None, None

		for f_i in range(self.features):
			feature_hist = node.hist[f_i]
			hist_cumsum = np.cumsum(feature_hist, axis=0)

			for i in range(self.bins - 1):
				left_hist = hist_cumsum[i]
				right_hist = hist_cumsum[-1] - hist_cumsum[i]

				total_samples = left_hist[-1] + right_hist[-1]

				if left_hist[2] < self.min_samples_node or right_hist[2] < self.min_samples_node:
					continue

				left_mean = left_hist[0]/left_hist[2]
				right_mean = right_hist[0]/right_hist[2]

				left_mse = left_hist[1]/left_hist[2] - left_mean**2
				right_mse = right_hist[1]/right_hist[2] - right_mean**2

				final_error = (left_hist[2]/total_samples) * left_mse + (right_hist[2]/total_samples) * right_mse

				if final_error < best_error:
					best_error = final_error
					split_feature = f_i
					split_bin = i

		return best_error, split_feature, split_bin

	def _split_node(self, node: DecisionTreeRegressorNode):
		_, split_feature, split_bin = self._find_best_split(node)

		if split_feature == None:
			return None, None, None

		parent_actives = node.active_samples

		# Getting the threshold for a given split
		bin_index = self.hist_intervals[split_bin + 1]
		bin_sup_limit_index = self.sorted_features[bin_index, split_feature]
		split_threshold = self.X[bin_sup_limit_index, split_feature]

		left_active_samples = np.zeros_like(parent_actives, dtype=bool)
		right_active_samples = np.zeros_like(parent_actives, dtype=bool)

		left_active_samples[self.sorted_features[:, split_feature][:bin_index]] = True
		right_active_samples[self.sorted_features[:, split_feature][bin_index:]] = True

		# Set to false all the observations that wasnt active in the parent node
		current_X_feat = self.X[:, split_feature]
		left_active_samples = (current_X_feat <= split_threshold) & parent_actives
		right_active_samples = (~(current_X_feat <= split_threshold)) & parent_actives

		# Build histogram for left and right nodes
		left_hist = self._build_hist(left_active_samples)
		right_hist = node.hist - left_hist

		return (left_hist, left_active_samples), (right_hist, right_active_samples), (split_feature, split_threshold)

	def _sup_fit(self, node: DecisionTreeRegressorNode, depth: int = 0):
		# Constraints defined by the hyperparameters. Limit tree growth. Avoids overfitting
		if depth >= self.max_depth or self.n_leaves >= self.max_n_leaves:
			self.n_leaves += 1
			return node

		left_node_info, right_node_info, split_info = self._split_node(node)
		# print(left_node_info, right_node_info)

		if left_node_info is not None and right_node_info is not None:
			node.split_feature = split_info[0]
			node.split_threshold = split_info[1]


			self.n_leaves += 1
			left_node = DecisionTreeRegressorNode(left_node_info[0], left_node_info[1])
			right_node = DecisionTreeRegressorNode(right_node_info[0], right_node_info[1])

			node.left = self._sup_fit(left_node, depth + 1)
			node.right = self._sup_fit(right_node, depth + 1)

		return node

	def fit(self, X: np.array, y: np.array):
		self.X, self.y = X.copy(), y.copy()
		self.sorted_features = np.argsort(X, axis=0)

		self.observations, self.features = X.shape

		hist_step = self.observations//self.bins
		self.hist_intervals = np.linspace(0, self.observations, self.bins + 1).astype(int)

		root_active_samples = np.ones_like(self.y, dtype=bool)
		root_hist = self._build_hist(root_active_samples)

		self.root = DecisionTreeRegressorNode(root_hist, root_active_samples)

		self._sup_fit(self.root)

	def _sup_predict(self, x_i: np.array, node: DecisionTreeRegressorNode):
		# Using mean of the leaf as the output 
		if node.left == None and node.right == None:
			return node.hist[0, :, 0].sum() / node.hist[0, :, 2].sum()

		#Transversing the tree recursively one observation at a time
		if x_i[node.split_feature] >= node.split_threshold:
			return self._sup_predict(x_i, node.right)
		else:
			return self._sup_predict(x_i, node.left)


	def predict(self, X: np.array):
		#Predictions are made using a for loop. Maybe there is a way to transverse the tree faster
		predictions = np.array(
			[self._sup_predict(X[i], self.root) for i in range(X.shape[0])]
		)

		return predictions

	def _in_order(self, node: DecisionTreeRegressorNode):
		# Prints in order tree
		if node == None:
			return

		self._in_order(node.left)
		if node.left == None and node.right == None:
			print(f"Size:{node.active_samples.sum()}| Leaf")
		else:
			print(f"Size:{node.active_samples.sum()}| Feat:{node.split_feature} | Threshold: {node.split_threshold}")
		self._in_order(node.right)

if __name__ == "__main__":

	# Synthetic data to test implementation (function made by Gemini) ------------------
	def make_regression_dataset(n_samples=10000, n_features=3, noise=0.1, seed=42):
	    rng = np.random.default_rng(seed)
	    
	    # Features
	    X = rng.uniform(-5, 5, size=(n_samples, n_features))
	    
	    # Nonlinear target (good for testing trees)
	    y = (
	        np.sin(X[:, 0]) +
	        0.5 * X[:, 1]**2 -
	        0.3 * X[:, 2] +
	        noise * rng.normal(size=n_samples)
	    )
	    
	    return X, y
	# ---------------------------------------------------------------------

	X, y = make_regression_dataset(3000, 4)
	X_train, X_test, y_train, y_test = X[:2400], X[2400:], y[:2400], y[2400:]

	params = {
		'bins': 255,
		'max_n_leaves': 100,
		'min_samples_node': 30,
		'max_depth': 10
	}

	dcr = DecisionTreeRegressor(**params)
	dcr.fit(X_train, y_train)
	dcr._in_order(dcr.root)

	predictions = dcr.predict(X_train)
	train_rsme = np.sqrt(np.mean((y_train - predictions)**2))

	ss_res = np.sum((y_train - predictions)**2)
	ss_tot = np.sum((y_train - np.mean(y_train))**2)
	print("R^2 train:", 1-(ss_res/ss_tot))
	print("RSME train:", train_rsme)

	predictions = dcr.predict(X_test)
	test_rsme = np.sqrt(np.mean((y_test - predictions)**2))

	ss_res = np.sum((y_test - predictions)**2)
	ss_tot = np.sum((y_test - np.mean(y_test))**2)
	print("R^2 test:", 1-(ss_res/ss_tot))
	print("RSME test:", test_rsme)