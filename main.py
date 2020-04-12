import numpy as np
import os
import cv2
import argparse
from skimage import io

def main():
    data_dir = os.path.dirname(__file__) + './cat-dataset/CAT_00'
    image_paths, coordinate_paths = get_cats(data_dir)
    images, coords = load_data(image_paths, coordinate_paths)
    print(len(images))

def load_data(image_paths, coordinate_paths):
    images = []
    for filepath in image_paths:
        im = io.imread(filepath)
        images.append(im)
    coords_list = []
    for cat_path in coordinate_paths:
        with open(cat_path) as f:
            coords = f.read().split()
            coords_list.append(coords)
            f.close()
    return images, coords_list

def get_cats(search_path):
    image_paths = []
    coordinate_paths= []
    for subdir, dirs, files in os.walk(search_path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".jpg"):
                image_paths.append(filepath)
            if filepath.endswith(".jpg.cat"):
                coordinate_paths.append(filepath)
    return image_paths, coordinate_paths

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # arguments = parser.parse_args()
    main()