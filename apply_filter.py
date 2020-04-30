import argparse
from cv2 import imread
import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from model import Model
import tensorflow as tf
from add_filter import ashhat_filter
from add_bow import bow_filter


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's apply some SnapCat filters!")
    
    parser.add_argument(
        '--image',
        default=None,
        help='File path of cat image.')

    parser.add_argument(
        '--filter',
        required=True,
        choices=['ashhat', 'bow'],
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

    # Load and preprocess image
    filepath = ARGS.image
    original_image = imread(filepath)
    img, scaling_factor = preprocess(original_image)

    # Load model 
    model = Model(224,224)
    model.compile_model()
    model.load_trained_model()
    
    # Run image through model
    predicted = model.model.predict(img)
    predicted = tf.reshape(predicted, (9,2))
    predicted = predicted.numpy()

    # Convert coordinate locations back to original image size 
    coords = np.floor(predicted * scaling_factor).astype(np.int)

    # Apply filter

    if ARGS.filter == 'ashhat':
        print("Applying Ash hat filter!")
        ashhat_filter(original_image, coords)      

    if ARGS.filter == 'bow':
        print("Applying bow filter!")
        bow_filter(original_image, coords)


# Make arguments global
ARGS = parse_args()

main()
