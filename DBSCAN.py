import numpy as np
from BaseModel import BaseModel
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

NOT_VISITED = -1
VISITED = 0

class DBSCAN(BaseModel):
	def __init__(self, min_points: int = 4, eps: float = 0.5):
		self.min_points = min_points
		self.eps = eps

	#Needed to access pairwise values from flat pdist (flattened triangle distance matrix)
	def _get_distance(self, i, j):
	    if i == j:
	        return 0.0
	    if i > j:
	        i, j = j, i
	    k = self.observations * i - i*(i+1)//2 + (j - i - 1)
	    return self.dist_mat[k]

	def _neighborhood(self, point):
		#Checks inside the triangular distance matrix for neighborhood (points with distance less or equal to eps)
		neighborhood = [i for i in range(self.observations) if self._get_distance(point, i) <= self.eps]
		return neighborhood

	def _expand(self, point, neighborhood):
		#For all the point inside the neigborhood, get their neighborhoods and add to cluster if n > min_points
		for p in neighborhood:
			if self.labels[p]  < 1:
				self.labels[p] = self.current_cluster
				sub_neighborhood = self._neighborhood(p)

				if len(sub_neighborhood) >= self.min_points:
					neighborhood.extend(sub_neighborhood)


	def fit_predict(self, X: np.array) -> np.array:
		self.observations, self.features = X.shape

		#All observations start as NOT_VISITED
		self.labels = -np.ones(self.observations)

		#Calculates a triangular distance matrix for the dataset points
		self.dist_mat = pdist(X)

		#Current cluster starts at 1. Noise is labeled as 0
		self.current_cluster = 1

		#Loops through all the points until each one is noise or assigned to a cluster
		for p in range(self.observations):
			if self.labels[p] == NOT_VISITED:
				self.labels[p] = VISITED
			else:
				continue

			neighborhood = self._neighborhood(p)

			# If the point is a core point, create a cluster and start expanding
			if len(neighborhood) >= self.min_points:
				self.labels[p] = self.current_cluster
				self._expand(p, neighborhood)
				self.current_cluster += 1

		return self.labels


if __name__ == "__main__":
	N_POINTS = 500
	np.random.seed(42)

	# Circle inside circle dataset
	X_circles = np.column_stack([[np.sin(x) for x  in np.linspace(0, 2*np.pi, N_POINTS//4)], [np.cos(x) for x  in np.linspace(0, 2*np.pi, N_POINTS//4)]])
	X_circles = np.concatenate([X_circles, -X_circles]) #Full circle
	X_circles = np.concatenate([X_circles, X_circles*1.8]) # Bigger circle
	X_circles += np.random.normal(0,0.07,(N_POINTS, 2)) # Noise

	#Moon dataset
	X_moon = np.random.uniform(0, np.pi, N_POINTS)

	X_moon = np.array([
		X_moon,
		np.sin(X_moon) + np.random.normal(0.7, 0.2, N_POINTS)
	]).T

	X_moon[:int(N_POINTS/2),:] -= 1.7
	X_moon[:int(N_POINTS/2),:] *= -1
	X_moon[:int(N_POINTS/2),1] -= 0.4

	#Normal clustering dataset w/ random noise observations sampled from uniform dist
	X_clusters = np.concatenate([
		np.random.normal(0.7,0.3,(N_POINTS//4, 2)),
		np.random.normal(2.1,0.3,(N_POINTS//4, 2)),
		np.random.normal(-0.7,0.3,(N_POINTS//4, 2)),
		np.random.normal(-2.1,0.3,(N_POINTS//4, 2)),
		np.random.uniform(-3, 3, (50, 2)) # noise
	])

	#DBSCAN & Plotting

	dbs = DBSCAN(min_points = 5, eps=0.3)
	datasets = [("Cicles", X_circles), ("Two moons", X_moon), ("4 Clusters w/noise", X_clusters)]
	n_ds = len(datasets)

	fig, ax = plt.subplots(1, n_ds, figsize=(12,6))
	fig.suptitle(f"DBSCAN algorithm w/ min_points = {dbs.min_points} and eps = {dbs.eps}")

	for i, dataset in enumerate(datasets):
		labels = dbs.fit_predict(dataset[1])

		ax[i].set_title(dataset[0])
		ax[i].scatter(dataset[1][:, 0], dataset[1][:, 1], c=labels)

	plt.show()

	#---------------------------------------------------