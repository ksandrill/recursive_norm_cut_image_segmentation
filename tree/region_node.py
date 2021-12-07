import numpy as np


class RegionNode:
    def __init__(self, image: np.ndarray):
        self.region = image
        self.children = [None, None]

    def get_region_size(self) -> int:
        return self.region[self.region > 0].size

    def get_region_color_tolerance(self) -> int:
        regionData = self.region[self.region > 0]
        return np.max(regionData) - np.min(regionData)

    def is_leaf(self) -> bool:
        return not any(self.children)
