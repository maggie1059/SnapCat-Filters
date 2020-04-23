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
    for i in range(7):
        data_dir = os.path.dirname(__file__) + './cat-dataset' + str(i)
        image_paths, coordinate_paths = get_cats(data_dir)
        print("retrieved paths")
        images_orig, coords_orig, max_x, max_y = load_data(image_paths, coordinate_paths)
        print("loaded images")
        coords_orig = parse_coordinates(coords_orig)
        # train_images, test_images, train_coords, test_coords = train_test_split(images_orig, coords_orig)
        print("parsed coords and split data")

        # this stuff will be inside a for loop for each epoch
        images = np.copy(images_orig)
        coords = np.copy(coords_orig)
        all_ims = np.zeros((len(images), 224, 224))
        all_coords = np.zeros(coords.shape)
        for i in range(len(images)):
            image = images[i]
            coord = coords[i]
            # image, coord = mirror1(image, coord)
            # print("done with mirror")
            max_x = max(image.shape)
            max_y = max_x
            image, coord = pad_image(image, coord, max_x, max_y)
            # print("done with padding")
            image, coord = resize_image(image, coord, 224, 224)
            # print("done with preprocessing")
            all_ims[i] = image
            all_coords[i] = coord
            # save_image(image, coord, i)
            # print("saved")
            # print(i)
        save_data(all_ims, all_coords, i)
    
def save_data(images, coords, folder):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    coords = coords[indices]
    num_test = int(len(images) * 0.2)
    test_images = images[:num_test]
    train_images = images[num_test:]
    test_coords = coords[:num_test]
    train_coords = coords[num_test:]

    np.save('./processed_test_imgs/img' + str(folder), test_images)
    np.save('./processed_test_coords/coord'+ str(folder), test_coords)
    
    np.save('./processed_train_imgs/img'+ str(folder), train_images)
    np.save('./processed_train_coords/coord'+ str(folder), train_coords)

def save_image(image, coord, filenum):
    np.save('./processed/' + str(filenum), image)
    np.save('./processed/' + str(filenum) + 'c', coord)

def load_data(image_paths, coordinate_paths):
    images = []
    max_x = 0
    max_y = 0
    for filepath in image_paths:
        im = imread(filepath)
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
    new_images = np.zeros((len(images), max_y, max_x))
    for i in range(len(images)):
        print(i)
        x_diff = max_x - images[i].shape[1]
        y_diff = max_y - images[i].shape[0]
        top = int(np.floor(np.random.rand()*y_diff))
        bottom = y_diff - top
        left = int(np.floor(np.random.rand()*x_diff))
        right = x_diff-left
        coords[i,:,0] += left
        coords[i,:,1] += top
        new_images[i,top:max_y-bottom,left:max_x-right] = images[i]
        # new_image = np.pad(images[i], ((top, bottom),(left, right)), 'constant', constant_values=(0,0))
        # new_images.append(new_image)
        # print(images[i].shape)
    return new_images, coords

def pad_image(image, coord, max_x, max_y):
    new_image = np.zeros((max_y, max_x))
    x_diff = max_x - image.shape[1]
    y_diff = max_y - image.shape[0]
    top = int(np.floor(np.random.rand()*y_diff))
    bottom = y_diff - top
    left = int(np.floor(np.random.rand()*x_diff))
    right = x_diff-left
    coord[:,0] += left
    coord[:,1] += top
    new_image[top:max_y-bottom,left:max_x-right] = image
    return new_image, coord

def mirror(images, coords, mirror_pct=0.3):
    for i in range(len(images)):
        if np.random.rand() < mirror_pct:
            images[i] = np.fliplr(images[i])
            coords[i,:,0] = images[i].shape[1] - coords[i,:,0]
    return images, coords

def mirror1(image, coord, mirror_pct=0.3):
    if np.random.rand() < mirror_pct:
        image = np.fliplr(image)
        coord[:,0] = image.shape[1] - coord[:,0]
    return image, coord

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

def resize_image(image, coord, new_height, new_width):
    height = image.shape[0]
    width = image.shape[1]
    scale_width = new_width/float(width)
    scale_height = new_height/float(height)

    new_image = resize(image, (new_height, new_width))
    coord[:, 0] *= scale_width
    coord[:,0] = coord[:,0]
    coord[:,1] *= scale_height
    coord[:,1] = coord[:,1]
    return np.array(new_image), coord.astype(np.int)

if __name__ == '__main__':
    main()