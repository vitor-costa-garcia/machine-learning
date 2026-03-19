import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
	def __init__(self, n_clusters: int, max_iter: int, tol: float):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.centroids = list()

	def fit_predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Kmeans.fit_predict(X[n:d]) -> np.ndarray[n:1]
		"""
		rows, cols = X.shape

		assert self.n_clusters <= rows, "O número de clusters deve ser menor ou igual ao número de observações do conjunto de dados" 

		#Escolhe k observações como centróides inicias
		indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
		self.centroids = X[indices, :]

		for i in range(self.max_iter):
			# Distâncias euclideanas de cada ponto para cada centróide
			euclidean_distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids]).T

			# Classifica os pontos baseado no cluster mais próximo
			labels = np.argmin(euclidean_distances, axis=1)

			# Limpa a lista de centróides
			old_centroids = self.centroids.copy()

			# Calculando os novos centróides
			self.centroids = np.vstack([np.mean(X[labels == k], axis=0) for k in range(self.n_clusters)])

			#Verificando deslocamento dos centroides para early-stopping
			eucl_dist_new_old_centroids = np.linalg.norm(old_centroids - self.centroids, axis=1)
			if np.all(eucl_dist_new_old_centroids <= self.tol):
				#print("|| EARLY STOPPING NA {i+1}º ITERAÇÃO ||")
				break
		
		return labels

	def wcss_(self, X: np.ndarray, max_k: int):
		rows, cols = X.shape

		assert max_k <= rows, "O número de clusters deve ser menor ou igual ao número de observações do conjunto de dados" 

		#Salvando as variáveis anteriores para reutilizar o método fit_transform
		previous_centroids = self.centroids.copy()
		previous_n_clusters = self.n_clusters

		wcss = list()

		for k_iter in range(1, max_k+1):
			self.n_clusters = k_iter
			labels = self.fit_predict(X)
			wcss_k = 0.0

			for ci in range(self.n_clusters):
				wcss_k += np.sum(np.linalg.norm(X[labels == ci] - self.centroids[ci,:], axis=0), axis=0)

			wcss.append(wcss_k)

		self.centroids = previous_centroids
		self.n_clusters = previous_n_clusters
		return wcss

	def sillhouete_score_(self, X: np.ndarray, max_k: int):
		rows, cols = X.shape

		assert max_k <= rows, "O número de clusters deve ser menor ou igual ao número de observações do conjunto de dados" 

		#Salvando as variáveis anteriores para reutilizar o método fit_transform
		previous_centroids = self.centroids.copy()
		previous_n_clusters = self.n_clusters

		dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

		# Salvando grupos de observações em clusters
		sillhouete_score = list()

		for k_iter in range(2, max_k+1):
			print(f"Sillhouete Score for K={k_iter}...")
			self.n_clusters = k_iter
			labels = self.fit_predict(X)

			values, counts = np.unique(labels, return_counts=True)
			ci_n = counts[labels]

			# clusters_obs = list()
			# for cluster_i in range(k_iter):
			# 	clusters_obs.append(X[labels == cluster_i])
			# Distância média entre o ponto i e os demais pontos do mesmo cluster
			a_i = []

			for obs in range(rows):
				sum_dist = np.sum(dist_matrix[obs, labels == labels[obs]])
				a_i.append(sum_dist)

			a_i = np.array(a_i) / (ci_n - 1)
				
			# Menor distância média entre o ponto i e os pontos de outros clusters
			b_i = []

			for obs in range(rows):
				temp_bi = []
				for cluster_i in range(k_iter):
					if cluster_i == labels[obs] or counts[cluster_i] == 0:
						continue

					sum_dist = np.sum(dist_matrix[obs, labels == cluster_i]) / counts[cluster_i]
					temp_bi.append(sum_dist)

				b_i.append(np.min(temp_bi))

			b_i = np.array(b_i)

			denom = np.maximum(a_i, b_i)

			sillhouete_score.append(np.mean((b_i - a_i) / denom))

		self.centroids = previous_centroids
		self.n_clusters = previous_n_clusters
		return sillhouete_score

if __name__ == "__main__":
	kmeans = Kmeans(4, 20, 10e-5)

	#Gerando clusters
	X = np.concatenate([
	    np.random.normal(loc=0, scale=1, size=(100, 2)),
	    np.random.normal(loc=5, scale=1, size=(100, 2)),
	    np.random.normal(loc=-5, scale=2, size=(100, 2)),
	    np.random.normal(loc=12, scale=2, size=(100, 2)),
	])

	labels = kmeans.fit_predict(X)

	fig, ax = plt.subplots()

	ax.scatter(X[:, 0], X[:, 1], c=labels) #Observações
	ax.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c='black', marker='*')# Centróides
	ax.set_title("K-means | Conjunto 2D gerado a partir de 4 distribuições normais")
	plt.show()

	# wcss = kmeans.wcss_(X, 12)
	# fig, ax = plt.subplots()
	# ax.set_title("Gráfico para o método do cotovelo WCSS")
	# ax.set_xlabel("K")
	# ax.set_ylabel("WCSS")
	# ax.set_xticks([i for i in range(1, 13)])
	# ax.plot([i for i in range(1, 13)], wcss, marker='s', markersize=10, linestyle='dashed')
	# plt.show()

	sillhouete_score = kmeans.sillhouete_score_(X, 12)
	fig, ax = plt.subplots()
	ax.set_title("Gráfico Sillhouete score")
	ax.set_xlabel("K")
	ax.set_ylabel("Score")
	ax.set_xticks([i for i in range(2, 13)])
	ax.plot([i for i in range(2, 13)], sillhouete_score, marker='s', markersize=10, linestyle='dashed', markerfacecolor='yellow', markeredgecolor='red')
	plt.show()