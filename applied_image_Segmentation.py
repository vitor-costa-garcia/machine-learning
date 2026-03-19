from Kmeans import Kmeans
from GaussianMixtureModel import GaussianMixtureModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.gridSampler import gridSampler

image_file = 'images/dog.png'

img = Image.open(image_file)

img_arr = np.array(img)

rows, cols, _ = img_arr.shape

RGB_matrix = np.ones((rows*cols, 3))
RGB_matrix[:, 0] = img_arr[:,:,0].flatten()
RGB_matrix[:, 1] = img_arr[:,:,1].flatten()
RGB_matrix[:, 2] = img_arr[:,:,2].flatten()

#Kmeans algorithm instance
# kmeans = Kmeans(1, 1000, 10e-4)

#Plotting some examples -----------------------------------

# fig, ax = plt.subplots(2,3, figsize=(12,7))

# fig.suptitle(f"K-means image segmentation example")

# ax[0][0].set_title("Original image")
# ax[0][0].imshow(img_arr)

# cmap = 'Accent'

# for k in range(2, 7): # Ks from 2 to 6
# 	i = (k-1) // 3
# 	j = (k-1) - i * 3

# 	kmeans.n_clusters = k
# 	labels = kmeans.fit_predict(RGB_matrix).reshape(rows, cols)

# 	ax[i][j].set_title(f"K = {k}")
# 	ax[i][j].imshow(labels, cmap=cmap)

# plt.savefig("images/dog_img_segmentation.png")
# plt.show()

# ---------------------------------------------------------

#Sillhouete Score for cluster hyperparameter choice ------
# This sillhouete score is very memory hungry! Keep N_SAMPLED_POINTS under 10k.

# MAX_K_SS = 6
# N_SAMPLED_POINTS = 5000

# rows, cols, _ = img_arr.shape

# ss = kmeans.sillhouete_score_(RGB_matrix[gridSampler(rows, cols, N_SAMPLED_POINTS)], MAX_K_SS)
# fig, ax = plt.subplots()
# ax.set_title("Gráfico Sillhouete Score")
# ax.set_xlabel("K")
# ax.set_ylabel("Score")
# ax.set_xticks([i for i in range(2, MAX_K_SS+1)])
# ax.plot([i for i in range(2, MAX_K_SS+1)], ss, marker='s', markersize=10, linestyle='dashed')
# plt.show()

# ---------------------------------------------------------

# ---------------------------------------------------------

#WCSS for Elbow method cluster hyperparameter choice ------

# MAX_K_WCSS = 6
# wcss = kmeans.wcss_(RGB_matrix, MAX_K_WCSS)
# fig, ax = plt.subplots()
# ax.set_title("Gráfico para o método do cotovelo WCSS")
# ax.set_xlabel("K")
# ax.set_ylabel("WCSS")
# ax.set_xticks([i for i in range(1, MAX_K_WCSS+1)])
# ax.plot([i for i in range(1, MAX_K_WCSS+1)], wcss, marker='s', markersize=10, linestyle='dashed')
# plt.show()

# ---------------------------------------------------------

# Using optimal K = 4 for segmentation of 'dog.png' -------

# kmeans.n_clusters = 4
# labels = kmeans.fit_predict(RGB_matrix).reshape(rows, cols)

# values, counts = np.unique(labels, return_counts=True)
# background_class = values[np.argmax(counts)]

# mask = labels != background_class

# dog_img = img_arr.copy()
# dog_img[~mask] = 0

# background_img  = img_arr.copy()
# background_img[mask] = 0

# fig, ax = plt.subplots(1,3, figsize=(12,5))

# fig.suptitle(f"K-means image segmentation example")

# ax[0].set_title("Original image")
# ax[0].imshow(img_arr)

# ax[1].set_title("Dog")
# ax[1].imshow(dog_img)

# ax[2].set_title("Background")
# ax[2].imshow(background_img)

# plt.savefig("images/dog_img_segmentation_optimal.png")
# plt.show()

# --------------------------------------------------------

# GMM in image segmentation
fig, ax = plt.subplots(1, 4, figsize=(12,5))

fig.suptitle(f"GMM image segmentation example")

ax[0].set_title("Original image")
ax[0].imshow(img_arr)

cmap = 'Accent'

for k in range(2, 5): # Ks from 2 to 6
	gmm = GaussianMixtureModel(n_components=k, tol=10e-3)

	gmm.n_components = k
	labels = gmm.fit_transform(RGB_matrix).reshape(rows, cols)

	ax[k-1].set_title(f"K = {k}")
	ax[k-1].imshow(labels)

plt.savefig("images/dog_img_segmentation_gmm.png")
plt.show()

# ----------------------------------------------
