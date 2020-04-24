import numpy as np
import os
import cv2
import argparse
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # np.save('./processed_test_imgs/img' + str(folder), test_images)
    # np.save('./processed_test_coords/coord'+ str(folder), test_coords)
    
    # np.save('./processed_train_imgs/img'+ str(folder), train_images)
    # np.save('./processed_train_coords/coord'+ str(folder), train_coords)

    img_path = 'processed_train_imgs/img0.npy'
    coord_path = 'processed_train_coords/coord0.npy'
    images = np.load(img_path)
    coords = np.load(coord_path)
    model = Model(224, 224)
    model.compile_model()
    model.train_model()
    for img_index in range(10, 13):
        model.visualize_points(images[img_index], coords[img_index], img_index)


def load_data(image_paths, coordinate_paths):
    images = []
    max_x = 0
    max_y = 0
    for filepath in image_paths:
        im = io.imread(filepath)
        im = rgb2gray(im)
        images.append(im)
        max_x = max(max_x, im.shape[1])
        max_y = max(max_y, im.shape[0])
    coords_list = []
    for cat_path in coordinate_paths:
        with open(cat_path) as f:
            coords = f.read().split()
            coords = [int(x) for x in coords]
            coords_list.append(coords)
            f.close()
    return np.array(images), coords_list, max_x, max_y

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # arguments = parser.parse_args()
    main()
