import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import cv2

#b. Write a function spectral clustering(x, k)
def spectral_clustering(x, k, r):
    # Construct a symmetric n*n matrix
    y = pdist(x, metric='sqeuclidean')
    z = squareform(y)
    matrix_a = np.exp(-r * z)

    # Compute the Laplacian matrix
    row_sum = matrix_a.sum(axis=1)
    matrix_d = np.diag(row_sum)
    matrix_l = matrix_d - matrix_a

    # Compute the eigenvalues and eigenvectors of L, and select eigenvectors Vk
    # corresponding to the k smallest eigenvalues of L.
    eigenvalues, eigenvectors = eigh(matrix_l, eigvals=(0, k - 1))

    # Use sklearn.cluster.KMeans to cluster the rows vi into clusters C1,...,Ck.
    kmeans = KMeans(n_clusters=k, random_state=10).fit_predict(eigenvectors)

    # Generate the clustering output
    return kmeans

if __name__ == '__main__':
    # a.Generate a noisy, synthetic data set
    x = datasets.make_circles(n_samples=1500, factor=.5, noise=.05)

    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

    # c.Use sklearn.cluster.KMeans directly to compute an alternative clustering for the two
    # dimensional circles data set generated above.
    kmeans_c = KMeans(n_clusters=2, random_state=10).fit(x[0])

    # Generate a scatter plot of the clusters, where the points are colored by the
    # cluster they belong to.
    ax1.set_title('KMeans')
    ax1.scatter(x[0][:, 0], x[0][:, 1], c=kmeans_c.labels_.astype(float))

    # d.Use your function spectral clustering(x, k) to compute the clustering for the two
    # dimensional circles data set generated above with k = 2 and different values of gamma.
    gamma = 27
    spectral_clustering = spectral_clustering(x[0], 2, gamma)

    # Generate a scatter plot of the clusters, , where the points are colored by the
    # cluster they belong to.
    ax2.set_title('Spectral Clustering gamma = %d' % gamma)
    ax2.scatter(x[0][:, 0], x[0][:, 1], c=spectral_clustering.astype(float))
    plt.show()
    '''

    # e.Load the 81  121 image seg.jpg
    img = cv2.imread('seg.jpg', 0)
    img_arr = img.reshape(len(img)*len(img[0]), 1)

    # Perform k-means with k = 2 and visualize the segmented images
    kmeans_e = KMeans(n_clusters=2, random_state=10).fit_predict(img_arr)
    img_kmeans = np.asmatrix(kmeans_e.reshape(len(img), len(img[0])))
    plt.subplot(1, 2, 1)
    plt.title('KMeans', fontsize=10)
    plt.imshow(img_kmeans, cmap='gray_r')

    # Perform spectral clustering with k = 2 and visualize the segmented images
    gamma = 1
    spectral_clustering = spectral_clustering(img_arr, 2, gamma)
    img_spectral_clustering = np.asmatrix(spectral_clustering.reshape(len(img), len(img[0])))
    plt.subplot(1, 2, 2)
    plt.title('Spectral Clustering gamma = {:.4f}'.format(gamma), fontsize=10)
    plt.imshow(img_spectral_clustering, cmap='gray_r')
    plt.show()
