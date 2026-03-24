import numpy as np

class DecisionTreeClassifierNode:
	def __init__(self, hist:np.array, active_samples: np.array):
		# Histogram (count classes per bin) and active samples for given node
		self.hist = hist
		self.active_samples = active_samples

		# Split info for predictions
		self.split_feature = None
		self.split_threshold = None

		# Child nodes
		self.left = None
		self.right = None

class DecisionTreeClassifier:
	def __init__(self, bins, max_n_leaves, min_samples_node, max_depth):
		self.bins = bins
		self.max_n_leaves = max_n_leaves
		self.min_samples_node = min_samples_node
		self.max_depth = max_depth

		self.n_leaves = 1

	def _build_hist(self, active_samples:np.array):
		hist = list()
		# For each sorted feature,
		for f_i in range(self.features):
			# For each bin, i need to count each class
			sorted_idx = self.sorted_features[:, f_i]      # Global sorted order
			active_sorted = active_samples[sorted_idx]     # Which of them belong to this node
			y_sorted = self.y[sorted_idx]                  # labels in sorted order

			# Creates a np.ndarray with count of each class per bin per feature
			class_count_bin = list()
			for i in range(self.bins):
				#Histogram ith bin intervals (indexes)
				start = self.hist_intervals[i]
				end = self.hist_intervals[i + 1]

				# Only the y's inside the ith bin
				mask = active_sorted[start:end]

				# Count classes inside bin
				counts = np.bincount(
					y_sorted[start:end][mask],
					minlength=self.classes
				)

				class_count_bin.append(counts)

			#Add feature histogram to final hist
			hist.append(class_count_bin)

		hist = np.array(hist)
		return hist

	def _find_best_split(self, node: DecisionTreeClassifierNode):
		lowest_gini, split_bin, split_feature = 2, None, None

		# We want to minimize weighted gini impurity 
		for f_i in range(self.features):
			#Feature histogram cumulative bin sum. Optimization
			feature_hist = node.hist[f_i]
			hist_cumsum = np.cumsum(feature_hist, axis=0)

			for i in range(self.bins - 1):
				#Counts of each class for each split
				left_counts = hist_cumsum[i]
				right_counts = hist_cumsum[-1] - hist_cumsum[i]

				# Total observations inside each split
				left_total = left_counts.sum()
				right_total = right_counts.sum()

				#Constraint min_samples_node. Avoids overfitting
				if (left_total < self.min_samples_node) or (right_total < self.min_samples_node):
					continue 

				split_total = left_total + right_total

				#Gini impurity for each split
				left_gini = 1 - np.sum((left_counts/left_total)**2)
				right_gini = 1 - np.sum((right_counts/right_total)**2)

				# Weighted gini impurity. must be minimized
				split_gini = (left_total/split_total) * left_gini + (right_total/split_total) * right_gini

				#Minimizing gini
				if split_gini < lowest_gini:
					lowest_gini = split_gini
					split_bin = i
					split_feature = f_i

		# In case no split where minimum requirements are satistifed, returns (2, None, None)
		return (lowest_gini, split_bin, split_feature)

	def _split_node(self, node: DecisionTreeClassifierNode):
		# Find_best_split function tries every combination of bins x features for the best split
		# best split is measured by weighted gini impurity measure, which must be minimized
		# A low gini impurity measure means that both of the resulting groups from the slip are well segregated by class
		lowest_gini, split_bin, split_feature = self._find_best_split(node)

		# If no valid split is found, find_best_split returns 2
		if lowest_gini > 1:
			return None, None, None

		# Getting the threshold for a given split
		bin_index = self.hist_intervals[split_bin + 1]
		bin_sup_limit_index = self.sorted_features[bin_index, split_feature]
		split_threshold = self.X[bin_sup_limit_index, split_feature]

		parent_actives = node.active_samples

		# Set to true all the observations inside the respective split
		left_active_samples = np.zeros_like(node.active_samples, dtype=bool)
		right_active_samples = np.zeros_like(node.active_samples, dtype=bool)

		left_active_samples[self.sorted_features[:, split_feature][:bin_index]] = True
		right_active_samples[self.sorted_features[:, split_feature][bin_index:]] = True

		# Set to false all the observations that wasnt active in the parent node
		left_active_samples &= node.active_samples
		right_active_samples &= node.active_samples

		# Build histogram for left and right nodes
		left_hist = self._build_hist(left_active_samples)
		right_hist = node.hist - left_hist

		return (left_hist, left_active_samples), (right_hist, right_active_samples), (split_feature, split_threshold)

	def _sup_fit(self, node: DecisionTreeClassifierNode, depth: int = 0):
		# Constraints defined by the hyperparameters. Limit tree growth. Avoids overfitting
		if depth >= self.max_depth or self.n_leaves >= self.max_n_leaves:
			self.n_leaves += 1
			return node

		# Split_node function returns histogram and active samples for left and right new nodes.
		#If no valid split is found, end recursion
		left_node_info, right_node_info, split_info = self._split_node(node)

		if left_node_info and right_node_info:
			#Split node info for predictions
			node.split_feature = split_info[0]
			node.split_threshold = split_info[1]

			self.n_leaves += 1
			left_node = DecisionTreeClassifierNode(left_node_info[0], left_node_info[1]) 
			node.left = self._sup_fit(left_node, depth = depth + 1)

			right_node = DecisionTreeClassifierNode(right_node_info[0], right_node_info[1])
			node.right = self._sup_fit(right_node, depth = depth + 1)
		
		self.n_leaves += 1
		return node

	def fit(self, X: np.ndarray, y: np.array):
		# Copying data into a class member
		self.X = X.copy()
		self.y = y.copy()

		# Data shape
		self.classes = len(np.unique(y))
		self.observations, self.features = self.X.shape

		#Sorted indexes matrix for each feature
		self.sorted_features = np.argsort(X, axis=0)

		#Adding step to stop so the interval of the last bin is complete
		hist_step = self.observations//self.bins
		self.hist_intervals = np.arange(start = 0, stop = self.observations + hist_step, step = hist_step)

		#Root node parameters. Will propagate through the tree.
		root_active_samples = np.ones(self.observations, dtype=bool)
		root_hist = self._build_hist(root_active_samples)

		self.root = DecisionTreeClassifierNode(root_hist, root_active_samples)

		#Fit
		self._sup_fit(self.root)

	def _sup_predict(self, x_i: np.array, node: DecisionTreeClassifierNode):
		# Using mode of the leaf as the output 
		if node.left == None and node.right == None:
			return np.argmax(np.sum(node.hist[0], axis=0))

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


	def _in_order(self, node: DecisionTreeClassifierNode):
		# Prints in order tree
		if node == None:
			return

		self._in_order(node.left)
		if node.left == None and node.right == None:
			print(f"Size:{node.active_samples.sum()}| Leaf")
		else:
			print(f"Size:{node.active_samples.sum()}| Feat:{node.split_feature} | Threshold: {node.split_threshold}")
		self._in_order(node.right)

if __name__ == '__main__':

	# Synthetic data generation for decision tree classification (Function made by ChatGPT)----------------
	def make_tree_friendly_data(n_samples=1000, noise=0.1, seed=42):
	    np.random.seed(seed)

	    # 3 numerical features
	    X = np.random.uniform(-5, 5, size=(n_samples, 3))

	    x1 = X[:, 0]
	    x2 = X[:, 1]
	    x3 = X[:, 2]

	    # Decision-tree-friendly rules (axis-aligned splits)
	    y = np.zeros(n_samples, dtype=int)

	    # Rule 1
	    mask1 = (x1 > 1.5) & (x2 < -1)
	    y[mask1] = 1

	    # Rule 2
	    mask2 = (x1 <= 1.5) & (x3 > 2)
	    y[mask2] = 2

	    # Rule 3
	    mask3 = (x2 > 2)
	    y[mask3] = 3

	    # Add a little noise (5% label noise)
	    noise_idx = np.random.choice(n_samples, size=int(noise * n_samples), replace=False)
	    y[noise_idx] = np.random.randint(0, 4, size=len(noise_idx))

	    return X, y
	# -----------------------------------------------------------------------------------------------------

	#Testando árvore de classificação
	total_size = 2000
	X, y = make_tree_friendly_data(total_size)
	train_size = 1600
	test_size = total_size - train_size

	#Dividindo dados sintéticos em treino/teste
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	#Hiperparâmetros da árvore de decisão
	params = {
		'bins': 100,
		'max_n_leaves': 100,
		'min_samples_node': 20,
		'max_depth': 7
	}

	dtc = DecisionTreeClassifier(**params)
	dtc.fit(X_train, y_train)

	# Comparação acurácia de treinamento e teste
	predictions_train = dtc.predict(X_train)
	accuracy_train = np.sum(predictions_train == y_train)/train_size
	print("Accuracy train:", accuracy_train)

	predictions_test = dtc.predict(X_test)
	accuracy_test = np.sum(predictions_test == y_test)/test_size
	print("Accuracy test:", accuracy_test)