from time import *
from timeit import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import sparse
from scipy.sparse.linalg import svds


def get_neighbors(index: int, radius: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(((X - col) ** 2 + (Y - row) ** 2))
    mask = R < radius
    return (X[mask] + Y[mask] * width).astype(int), R[mask]


def compute_adjacency_matrix(flat_grayscale: np.ndarray,
                             image_size: tuple[int, int], r: float = 5., sigma_intensity: float = .02,
                             sigma_distance: float = 3.):
    vertex_number = len(flat_grayscale)
    image_h, image_w = image_size
    W = np.zeros((vertex_number, vertex_number))
    for i in range(vertex_number):
        neighbors, neighbors_distances = get_neighbors(i, r, image_h, image_w)
        color_similarity = np.exp(-np.abs(flat_grayscale[neighbors] - flat_grayscale[i]) / sigma_intensity)
        spatial_similarity = np.exp(-np.abs(neighbors_distances) / sigma_distance)
        weights = color_similarity + spatial_similarity
        W[i, neighbors] = weights
    return W


def weighted_diag_matrix(adjacency_matrix: np.ndarray) -> np.ndarray:
    D = np.sum(adjacency_matrix, axis=1)
    D = np.diag(D)
    return D


def get_eigen_vectors_from_second_smallest(image: np.ndarray, vector_number: int) -> np.ndarray:
    W = compute_adjacency_matrix(image.flatten().astype(float) / 255, image.shape[:2])
    D = weighted_diag_matrix(W)
    s_D = sparse.csr_matrix(D)
    s_W = sparse.csr_matrix(W)
    s_D_nhalf = np.sqrt(s_D).power(-1)
    L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
    _, _, eigen_vectors = svds(L, which='SM')
    return eigen_vectors[1:vector_number + 1, :]


def split_image(image: np.ndarray, eigen_vector: np.ndarray):
    split_point = np.mean(eigen_vector)
    mask = eigen_vector > np.mean(eigen_vector)
    mask = mask.reshape(image.shape)
    return image * mask, image * ~mask, split_point


def calc_cut_cost(eigen_vector: np.ndarray, image_h: int, image_w: int, split_point: float,
                  adjacency_matrix: np.ndarray, radius: float = 5):
    n_cut_cost = 0.0
    for i in range(len(eigen_vector)):
        neighbors, _ = get_neighbors(i, radius, image_h, image_w)
        for neighbor in neighbors:
            if eigen_vector[neighbor] > split_point >= eigen_vector[i] or eigen_vector[neighbor] <= split_point < \
                    eigen_vector[i]:
                n_cut_cost += adjacency_matrix[i, neighbor]
    return n_cut_cost / np.sum(adjacency_matrix)


def main():
    image = cv2.imread('../pictures/Lenna.png', 0)
    image = cv2.resize(image, (50, 50))
    vector_number = 5
    eigen_vectors = get_eigen_vectors_from_second_smallest(image,
                                                           vector_number=vector_number)  # from second smallest
    for eigen_vector in eigen_vectors:
        plt.figure()
        region1, region2,_ = split_image(image, eigen_vector)
        plt.subplot(1, 2, 1)
        skimage.io.imshow(region1)
        plt.subplot(1, 2, 2)
        skimage.io.imshow(region2)

        plt.show()


if __name__ == '__main__':
    t = timeit(lambda: main(), number=1)
    print(t)
