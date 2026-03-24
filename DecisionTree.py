import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeClassifierNode:
	def __init__(self, hist: np.array, active_samples: np.array):
		#self.feature_index = feature_index
		#self.cut = cut
		#self.X = X
		#self.y = y
		self.hist = hist
		self.active_samples = active_samples

	def redirect(self, X: np.array):
		#Base
		if self.observations:
			return self.observations

		if X[:, self.feature_index] >= self.cut:
			return redirect(self.right)

		else:
			return redirect(self.left)

	def split(self, metric: str = 'gini'):
		best_feature = -1
		best_gini = -1
		best_threshold = -1


class DecisionTreeClassifier:
	def __init__(self, max_n_leaves: int, max_height: int, min_samples_node: int, bins: int):
		# Hyperparameters
		self.max_n_leaves = max_n_leaves
		self.max_depth = max_height
		self.min_samples_node = min_samples_node
		self.bins = bins

		# Properties
		self.height = 0
		self.n_nodes = 0

	def _build_hist(self, active_samples):
		bins_counts = list()

		for col in range(self.cols):
			feat_bin_class_counts = list() 
			for i in range(self.bins):
				low, high = self.bins_intervals[i], self.bins_intervals[i+1]
				idx = self.sorted_idx[col][low:high]     # indexes of this bin
				idx = idx[active_samples[idx]]           # keep only active samples

				classes_count_bin = np.bincount(self.y[idx], minlength = len(self.classes))
				feat_bin_class_counts.append(classes_count_bin.reshape(1, -1))

			bins_counts.append(np.vstack(feat_bin_class_counts))

		#bin_count[feature][bin][class]
		return np.array(bins_counts)

	def _find_best_split(self, node: DecisionTreeClassifierNode):
		best_gini, best_feat, best_bin_split = (1, -1, -1)

		for col in range(self.cols):
			feat_hist = node.hist[col]
			cum_sum_bins = np.cumsum(feat_hist, axis=0)

			for i in range(self.bins-1):

				bin_split_idx = self.bins_intervals[i+1] #High index of each interval

				left_counts = cum_sum_bins[i]
				left_total = np.sum(left_counts)
				right_counts = cum_sum_bins[-1] - left_counts
				right_total = np.sum(right_counts)

				node_total = left_total + right_total

				if left_total == 0 or right_total == 0:
					continue

				gini_left = 1 - np.sum((left_counts/left_total)**2)
				gini_right = 1 - np.sum((right_counts/right_total)**2)

				# weighted mean
				gini_split = (left_total/node_total) * gini_left + (right_total/node_total) * gini_right

				if gini_split < best_gini:
					best_gini, best_feat, best_bin_split = gini_split, col, i+1

		return best_gini, best_feat, best_bin_split

	def _split_node(self, node: DecisionTreeClassifierNode):
		_, feat, bin_split = self._find_best_split(node)

		#need to calculate new left and new right!

		active_samples_left = node.active_samples.copy()
		active_samples_right = node.active_samples.copy()

		active_samples_left[:] = False
		active_samples_left[new_left_idx] = True

		active_samples_right[:] = False
		active_samples_right[new_right_idx] = True



	def fit(self, X: np.array, y: np.array):
		self.rows, self.cols = X.shape
		assert self.bins <= self.rows, "Número de bins inválido. A quantidade de bins deve ser menor ou igual ao número de observações."

		#storing data
		self.X = X
		self.y = y

		#compute count of class per bin
		self.classes = np.unique(y, axis=0)

		step = self.rows//self.bins
		self.bins_intervals = np.arange(start=0, stop=self.rows + step, step=step)

		self.sorted_idx = np.array([np.argsort(X[:, i]) for i in range(self.cols)])

		active_samples = np.full(self.rows, True, dtype=bool)
		self.bins_counts = self._build_hist(active_samples)

		self.head = DecisionTreeClassifierNode(self.bins_counts.copy(), active_samples)

		print(self._find_best_split(self.head))


	def predict(X: np.ndarray):
		pass

if __name__ == "__main__":
	N_POINTS = 400
	X = np.concatenate([
	    np.random.normal(loc=0, scale=100, size=(N_POINTS//4, 4)),
	    np.random.normal(loc=5, scale=100, size=(N_POINTS//4, 4)),
	    np.random.normal(loc=-5, scale=200, size=(N_POINTS//4, 4)),
	    np.random.normal(loc=12, scale=200, size=(N_POINTS//4, 4)),
	])
	y = np.random.choice(3, N_POINTS)
	dct = DecisionTreeClassifier(100, 100, 100, 30)
	dct.fit(X, y)
