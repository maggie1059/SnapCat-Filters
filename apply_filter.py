import argparse
import os
from cv2 import imread
import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from model import Model
import tensorflow as tf
from add_filter import ashhat_filter, james_filter
from add_bow import bow_filter
from add_dog_filter import dog_filter


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's apply some SnapCat filters!")

    parser.add_argument(
        '--image',
        default=None,
        help='File path of cat image or directory of cat images')

    parser.add_argument(
        '--filter',
        required=True,
        choices=['ashhat', 'bow','james', 'dog'],
        help='''Which SnapCat filter to add to image''')

    return parser.parse_args()

def preprocess(image):
    # Change to gray scale
    image = rgb2gray(image)

    # Pad to square (pad bottom and right side only)
    max_dim = np.max(image.shape)
    new_image = np.zeros((max_dim, max_dim))
    new_image[0:image.shape[0], 0:image.shape[1]] = image

    # Resize to 224 x 224
    new_image = resize(new_image, (224, 224))
    new_image = gray2rgb(new_image) / 255.

    scaling_factor = max_dim / 224

    return np.reshape(new_image, (1, 224, 224, 3)), scaling_factor



def main():

    # Load model
    model = Model(224,224)
    model.compile_model()
    model.load_trained_model()

    # Load and preprocess image(s)
    search_path = ARGS.image
    # If user passed in a directory
    if os.path.isdir(search_path):
        image_paths = []
        for subdir, dirs, files in os.walk(search_path):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".jpg") or filepath.endswith(".jpg"):
                    image_paths.append(filepath)
    # If user passed in a single image
    elif os.path.isfile(search_path): 
        image_paths = [search_path]
    # If the file path is invalid
    else:
        print("Error: Please input a valid file or directory path")
        return
    
    for i, image_path in enumerate(image_paths):
        original_image = imread(image_path)
        img, scaling_factor = preprocess(original_image)

        # Run image through model
        predicted = model.model.predict(img)
        predicted = tf.reshape(predicted, (9,2))
        predicted = predicted.numpy()

        # Convert coordinate locations back to original image size
        coords = np.floor(predicted * scaling_factor).astype(np.int)

        # Apply filter

        if ARGS.filter == 'ashhat':
            print("Applying Ash hat filter!")
            ashhat_filter(original_image, coords, output_image_path="cat_" + ARGS.filter + str(i) + ".png")

        if ARGS.filter == 'bow':
            print("Applying bow filter!")
            bow_filter(original_image, coords, output_image_path="cat_" + ARGS.filter + str(i) + ".png")

        if ARGS.filter == 'james':
            print("Applying James filter!")
            james_filter(original_image, coords, output_image_path="cat_" + ARGS.filter + str(i) + ".png")

        if ARGS.filter == 'dog':
            print("Applying dog filter!")
            dog_filter(original_image, coords, output_image_path="cat_" + ARGS.filter + str(i) + ".png")


# Make arguments global
ARGS = parse_args()

main()
