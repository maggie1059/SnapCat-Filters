import argparse
import cv2 as cv
from cv2 import imread
import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
from model import Model
import tensorflow as tf
from add_filter import ashhat_filter
from add_bow import bow_filter

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

    model = Model(224,224)
    model.compile_model()
    model.load_trained_model()

    cap = cv.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        img, scaling_factor = preprocess(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        predicted = model.model.predict(img)
        predicted = tf.reshape(predicted, (9,2))
        predicted = predicted.numpy()

        coords = np.floor(predicted * scaling_factor).astype(np.int)
        new_im = ashhat_filter(frame, coords)
        new_image_chan = np.copy(new_im[:,:,0])
        new_im[:,:,0] = new_im[:,:,2]
        new_im[:,:,2] = new_image_chan
        # new_im = bow_filter(frame, coords)
        # new_im *= 255
        # new_im = new_im.astype(np.int)

        cv.imshow('frame', new_im)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()
