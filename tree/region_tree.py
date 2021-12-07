import numpy as np

from n_cut.n_cut import get_eigen_vectors_from_second_smallest, split_image, compute_adjacency_matrix, calc_cut_cost
from tree.region_node import RegionNode


def is_okay_to_split_region(node: RegionNode, min_region_size: int):
    return node.get_region_size() >= min_region_size


class RegionTree:
    def __init__(self):
        self.root = None
        self.split_counter = 0

    def _split(self, node: RegionNode, min_region_size: int, cuts_number: int, thr: float) -> None:
        eigen_vectors = get_eigen_vectors_from_second_smallest(node.region,
                                                               vector_number=1)
        region1, region2, split_point = split_image(node.region, eigen_vectors[0])
        self.split_counter += 1
        restored_w = compute_adjacency_matrix(node.region.flatten().astype(float) / 255, node.region.shape[0:2])
        cut_cost = calc_cut_cost(eigen_vectors[0], node.region.shape[0], node.region.shape[1], split_point, restored_w)
        print('cut cost: ', cut_cost)
        node.children = RegionNode(region1), RegionNode(region2)
        if self.split_counter >= cuts_number:
            return
        if cut_cost <= thr:
            for i in range(len(node.children)):
                if is_okay_to_split_region(node.children[i], min_region_size):
                    # print(i + 1, ' child have size: ', node.children[i].get_region_size())
                    # print(i + 1, ' child have color tolerance ', node.children[i].get_region_color_tolerance())
                    if self.split_counter < cuts_number:
                        self._split(node.children[i], min_region_size, cuts_number=cuts_number, thr=thr)

    def make_tree(self, img: np.ndarray, min_region_size: int, cuts_number: int, thr=0.04) -> None:
        self.root = RegionNode(img)
        self._split(self.root, min_region_size, cuts_number=cuts_number, thr=thr)

    def _extract_leaves_to_list(self, node: RegionNode, region_list: list[np.ndarray]) -> None:
        if node.is_leaf():
            region_list.append(node.region)
        for child in node.children:
            if child is not None:
                self._extract_leaves_to_list(child, region_list)

    def extract_leaves_to_list(self) -> list[np.ndarray]:
        leaves_region_list = []
        self._extract_leaves_to_list(self.root, leaves_region_list)
        return leaves_region_list
