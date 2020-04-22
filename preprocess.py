import numpy as np
import os
import cv2
import argparse
from skimage import io
from cv2 import imread
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    data_dir = os.path.dirname(__file__) + './cat-dataset/CAT_00'
    image_paths, coordinate_paths = get_cats(data_dir)
    print("retrieved paths")
    images_orig, coords_orig, max_x, max_y = load_data(image_paths, coordinate_paths)
    print("loaded images")
    coords_orig = parse_coordinates(coords_orig)
    train_images, test_images, train_coords, test_coords = train_test_split(images_orig, coords_orig)
    print("parsed coords and split data")

    # this stuff will be inside a for loop for each epoch
    
    images = np.copy(train_images)
    coords = np.copy(train_coords)

    images, coords = mirror(images, coords)
    print("done with mirror")
    images, coords = pad_images(images, coords, max_x, max_y)
    print("done with padding")
    images, coords = resize_images(images, coords, 224, 224)
    print("done with preprocessing")
    save_data(images, coords)
    print("saved")
    
def save_data(images, coords):
    counter = 0
    for i in range(len(images)):
        np.save('./processed/' + counter, images[i])
        np.save('./processed/' + counter + 'c', coords[i])

def load_data(image_paths, coordinate_paths):
    images = []
    max_x = 0
    max_y = 0
    counter = 0
    for filepath in image_paths:
        print(counter)
        im = imread(filepath)
        im = rgb2gray(im)
        images.append(im)
        max_x = max(max_x, im.shape[1])
        max_y = max(max_y, im.shape[0])
        counter+=1
    coords_list = []
    for cat_path in coordinate_paths:
        with open(cat_path) as f:
            coords = f.read().split()
            coords = [int(x) for x in coords]
            coords_list.append(coords)
            f.close()
    return np.array(images), coords_list, max_x, max_y

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
    image_paths = sorted(image_paths)
    coordinate_paths = sorted(coordinate_paths)
    return image_paths, coordinate_paths

def parse_coordinates(coords_list):
    # discard any files with more than 9 coordinates
    # separate pairs of coordinates
    final_coords = []
    for i in coords_list:
        if i[0] == 9:
            coords = np.array(i[1:])
            coords = np.reshape(coords, (9,2))
            final_coords.append(coords)
    # final_coords = np.array(final_coords)
    return np.array(final_coords, dtype=np.float64)

# process before this so images + coords are both np arrays
def train_test_split(images, coords, test_split=0.2):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    coords = coords[indices]
    num_test = int(len(images) * test_split)
    test_images = images[:num_test]
    train_images = images[num_test:]
    test_coords = coords[:num_test]
    train_coords = coords[num_test:]
    return train_images, test_images, train_coords, test_coords

def pad_images(images, coords, max_x, max_y):
    # print(max_x, max_y)
    # max_x = 500
    # max_y = 500
    # for image, coord in zip(images,coords):
    new_images = []
    for i in range(len(images)):
        x_diff = max_x - images[i].shape[1]
        y_diff = max_y - images[i].shape[0]
        top = int(np.floor(np.random.rand()*y_diff))
        bottom = y_diff - top
        left = int(np.floor(np.random.rand()*x_diff))
        right = x_diff-left
        coords[i,:,0] += left
        coords[i,:,1] += top
        new_image = np.pad(images[i], ((top, bottom),(left, right)), 'constant', constant_values=(0,0))
        new_images.append(new_image)
        # print(images[i].shape)
    return np.array(new_images), coords

def mirror(images, coords, mirror_pct=0.3):
    for i in range(len(images)):
        if np.random.rand() < mirror_pct:
            images[i] = np.fliplr(images[i])
            coords[i,:,0] = images[i].shape[1] - coords[i,:,0]
    return images, coords

def resize_images(images, coords, new_height, new_width):
    height = images.shape[1]
    width = images.shape[2]
    scale_width = new_width/float(width)
    scale_height = new_height/float(height)
    new_images = []
    for i in range(len(images)):
        new_images.append(resize(images[i], (new_height, new_width)))
        coords[i,:, 0] *= scale_width
        coords[i,:,0] = coords[i,:,0]
        coords[i, :,1] *= scale_height
        coords[i,:,1] = coords[i,:,1]
    return np.array(new_images), coords.astype(np.int)


if __name__ == '__main__':
    main()