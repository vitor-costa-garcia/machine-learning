import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from PrincipalComponentsAnalysis import PrincipalComponentAnalysis

class KernelPrincipalComponentAnalysis():
    def __init__(self, n_components: int, kernel_function: str = 'rbf', gamma: float = 15):
        self.n_components = n_components
        self.kernel_function = kernel_function
        self.gamma = gamma

    def _kernel_rbf(self, X: np.ndarray):
        xm = X[:, np.newaxis, :]
        xn = X[np.newaxis, :, :]
        return np.exp(-self.gamma * (np.linalg.norm(xn-xm, axis=-1)**2))

    def fit_transform(self, X: np.ndarray):
        #Gerando a Gram Matrix, onde cada elemento a_ij=k(x_n, x_m)
        gram_matrix = self._kernel_rbf(X)

        # Centralizando a Gram Matrix
        row_means = np.mean(gram_matrix, axis=1, keepdims=True)
        col_means = np.mean(gram_matrix, axis=0, keepdims=True)
        total_mean = np.mean(gram_matrix)

        c_gram_matrix = gram_matrix - row_means - col_means + total_mean

        #Autovalores e autovetores da Gram Matrix centralizada
        eigvals, eigvecs = np.linalg.eigh(c_gram_matrix)
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, axis=1)

        # Matriz com os n autovetores da Gram Matrix
        kpca_matrix = np.array(eigvecs[:,:self.n_components])

        #Clipando os valores (erro de precisão numérica...)
        clean_eigvals = np.maximum(eigvals, 0)

        #Normalizando os autovetores
        s = np.sqrt(clean_eigvals * N_POINTS)
        s = np.where(s > 1e-9, s, 1.0)
        s_subset = s[:self.n_components]
        alphas = kpca_matrix / s_subset

        #Projetando os dados
        result = np.dot(c_gram_matrix, alphas[:, :self.n_components])

        return result


N_POINTS = 500

if __name__=="__main__":
    #Seed para geração do dataset
    np.random.seed(2)

    #Gerando "luas"
    xn = np.random.uniform(0, np.pi, N_POINTS)

    X = np.array([
        xn,
        np.sin(xn) + np.random.normal(0.7, 0.2, N_POINTS)
    ]).T

    X[:int(N_POINTS/2),:] -= 1.7
    X[:int(N_POINTS/2),:] *= -1
    X[:int(N_POINTS/2),1] -= 0.4

    labels = np.concat([
        np.zeros(floor(N_POINTS/2)),
        np.ones(ceil(N_POINTS/2))
    ])

    kpca = KernelPrincipalComponentAnalysis(1, gamma=4)
    X_kpca = kpca.fit_transform(X)

    pca = PrincipalComponentAnalysis(1)
    X_pca = pca.fit_transform(X)

    # Plot comparação KPCA x PCA no dataset de duas luas
    fig, ax = plt.subplots(1,3, figsize=(12, 6))

    ax[0].set_title("Moon data")
    ax[0].scatter(X[:,0], X[:,1], c=labels)

    ax[1].set_title("Kernel PCA with RBF")
    ax[1].scatter(X_kpca, np.zeros(N_POINTS), c=labels, s=3, marker='*')

    ax[2].set_title("Common PCA")
    ax[2].scatter(X_pca, np.zeros(N_POINTS), c=labels, s=3, marker='*')

    plt.show()