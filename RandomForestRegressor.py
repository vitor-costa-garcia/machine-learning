import numpy as np
from BaggingRegressor import BaggingRegressor
from DecisionTreeRegressor import DecisionTreeRegressor
from DecisionTreeRegressor import DecisionTreeRegressorNode
from BaseModel import BaseModel

#What i want to do is create a child class from the decision tree regressor that changes the find_best_split() method.
class DecisionTreeRFRegressor(DecisionTreeRegressor):
	def __init__(self, p: int, bins: int, max_n_leaves: int, min_samples_node: int, max_depth: int):
		super().__init__(bins, max_n_leaves, min_samples_node, max_depth)
		self.p = p # Random number of parameters

	def _find_best_split(self, node: DecisionTreeRegressorNode):
		best_error, split_feature, split_bin = float("inf"), None, None

		if node.active_samples.sum() < 2*self.min_samples_node:
			return None, None, None

		#This is the only change i need to make on decision tree class
		#Chooses p random features to adjust
		usable_features = np.random.choice(self.features, self.p, replace=False)

		for f_i in usable_features:
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

class RandomForestRegressor(BaseModel):
	def __init__(self, n_trees: int, p: int, bootstrap_frac: float, params: dict):
		self.n_trees = n_trees
		self.bootstrap_frac = bootstrap_frac
		self.tree_params = params
		self.tree_params['p'] = p

	def fit(self, X: np.array, y: np.array):
		self.bgg = BaggingRegressor(
								DecisionTreeRFRegressor,
								self.n_trees,
								self.bootstrap_frac,
								self.tree_params
							  	)
		self.bgg.fit(X, y)

	def predict(self, X: np.array):
		return self.bgg.predict(X)

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
		'bins': 300,
		'max_n_leaves': 255,
		'min_samples_node': 20,
		'max_depth': 10
	}

	dcr = RandomForestRegressor(n_trees = 50, p = int(np.sqrt(X.shape[1])), bootstrap_frac=1, params = params)
	dcr.fit(X_train, y_train)

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