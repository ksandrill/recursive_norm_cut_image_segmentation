import os

import cv2
import skimage
from matplotlib import pyplot as plt

from tree.region_tree import RegionTree

PATH_TO_SAVE = 'OUTPUT'
PATH_TO_IMAGE = 'pictures/Lenna.png'

INPUT_RESIZE_SHAPE = (50, 50)
OUTPUT_RESIZE_SHAPE = (300, 300)
if __name__ == '__main__':
    image = cv2.imread(PATH_TO_IMAGE, 0)
    image = cv2.resize(image, INPUT_RESIZE_SHAPE)
    tree = RegionTree()
    tree.make_tree(image, min_region_size=900, color_tol=50, cuts_number=5)
    regions = tree.extract_leaves_to_list()
    new_dir_name = os.path.basename(PATH_TO_IMAGE).split('.')[0]
    path_to_dir = os.path.join(PATH_TO_SAVE, new_dir_name)
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    region_number = len(regions)
    plt.figure()
    for i, region in enumerate(regions):
        plt.subplot(1, region_number, i + 1)
        image = cv2.resize(region, OUTPUT_RESIZE_SHAPE)
        skimage.io.imshow(image)
        # cv2.imshow(str(i + 1) + '_segment', image)
        #cv2.waitKey(0)
        cv2.imwrite(os.path.join(path_to_dir, 'segment_' + str(i + 1) + '.png'), image)
    plt.show()

