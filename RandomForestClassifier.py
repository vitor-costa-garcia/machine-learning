import numpy as np
from BaggingClassifier import BaggingClassifier
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeClassifier import DecisionTreeClassifierNode
from BaseModel import BaseModel

#What i want to do is create a child class from the decision tree classifier that changes the find_best_split() method.
class DecisionTreeRFClassifier(DecisionTreeClassifier):
	def __init__(self, p: int, bins: int, max_n_leaves: int, min_samples_node: int, max_depth: int):
		super().__init__(bins, max_n_leaves, min_samples_node, max_depth)
		self.p = p # Random number of parameters


	def _find_best_split(self, node: DecisionTreeClassifierNode):
		lowest_gini, split_bin, split_feature = 2, None, None

		#For each split, we choose self.p random features to take into account for the split
		usable_features = np.random.choice(self.features, self.p, replace=False)
		# We want to minimize weighted gini impurity 
		for f_i in usable_features:
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

class RandomForestClassifier(BaseModel):
	def __init__(self, n_trees: int, p: int, bootstrap_frac: float, params: dict):
		self.n_trees = n_trees
		self.bootstrap_frac = bootstrap_frac
		self.tree_params = params
		self.tree_params['p'] = p

	def fit(self, X: np.array, y: np.array):
		self.bgg = BaggingClassifier(
								DecisionTreeRFClassifier,
								self.n_trees,
								self.bootstrap_frac,
								self.tree_params
							  	)
		self.bgg.fit(X, y)

	def predict(self, X: np.array):
		return self.bgg.predict(X)

if __name__ == "__main__":
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
		'bins': 400,
		'max_n_leaves': 100,
		'min_samples_node': 20,
		'max_depth': 7
	}

	dtc = RandomForestClassifier(50, 1, 1, params)

	print("Fit started...")
	dtc.fit(X_train, y_train)
	print("Fit ended.")

	# Comparação acurácia de treinamento e teste
	print("Prediction train started...")
	predictions_train = dtc.predict(X_train)
	accuracy_train = np.sum(predictions_train == y_train)/train_size
	print("Accuracy train:", accuracy_train)
	print("Prediction train ended.")

	print("Prediction test started...")
	predictions_test = dtc.predict(X_test)
	accuracy_test = np.sum(predictions_test == y_test)/test_size
	print("Accuracy test:", accuracy_test)
	print("Prediction test ended.")