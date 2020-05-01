import numpy as np
import os
import cv2
from cv2 import imread
import imutils
import argparse
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from model import Model
from preprocess import get_cats,load_data

def toast_filter(image,coord):

    #remove first element (in the case of size 19)
    coord = coord[1:]
    #reshape if necessary
    coord = coord.reshape(9,2)
    #use tips when points are out of bounds
    left_tip = coord[4,:]
    right_tip = coord[-2,:]

    #otherwise use base of ear
    left_out_ear = coord[3,:]
    right_out_ear = coord[-1,:]

    mouth = coord[2, :]

    #try using mouth point for face
    mouth_point = coord[2,:]
    toast = cv2.imread('filters/toast.png',cv2.IMREAD_UNCHANGED)

    #set hat angle using tips only if out of bounds
    #set width accordingly
    angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
    toast_width = np.linalg.norm(right_tip - left_tip)
    toast_width *= 1.8
    toast_height = mouth_point[1] - min(right_tip[1],left_tip[1])
    toast_height *= 2.2

    rotate_toast = imutils.rotate_bound(toast,angle*180/np.pi)

    old_height, old_width, channel = toast.shape

    toast = cv2.resize(rotate_toast, (int(1.3*toast_width),int(1.3*toast_height)))
    hw,hh,hc = toast.shape

    toast = np.array(toast, dtype = np.float64)
    toast /= 255.0

    new_chan = np.copy(toast[:,:,0])
    toast[:,:,0] = toast[:,:,2]
    toast[:,:,2] = new_chan

    image = np.array(image,dtype =np.float64)
    image /= 255.0

    new_image_chan = np.copy(image[:,:,0])
    image[:,:,0] = image[:,:,2]
    image[:,:,2] = new_image_chan

    y = int(mouth[1] - toast_height/1.3)
    x = int(mouth[0] - toast_width/1.6)

    for i in range(0,hw):       # Overlay the filter based on the alpha channel
        for j in range(0,hh):
            if toast[i,j,-1] != 0 and i+y >=0 and j+x >=0 and i+y < len(image) and j+x < len(image[0]):
                # print(i+left_tip[1], j+left_tip[0])
                # print(i,j)
                # image[i+left_tip[1],j+left_tip[0],:] = toast[i,j,:-1]
                # image[i+left_tip[1]-80,j+left_tip[0]-180,:] = toast[i,j,:-1]
                # y = j + mouth[0] - toast_height/2
                # x = i + mouth[1] - toast_width/2
                image[i+y,j+x,:] = toast[i,j,:-1]

    plt.imshow(image)
    plt.show()

data_dir = os.path.dirname(__file__) + './cat-dataset/small'
image_paths, coordinate_paths = get_cats(data_dir)
print("retrieved paths")
images_orig, coords_orig, max_x, max_y = load_data(image_paths, coordinate_paths)
print("loaded images")

images = list()
for path in range(len(image_paths)):
    images.append(imread(image_paths[path]))
coords = np.copy(coords_orig)

for ind in range(len(images)):
    image = images[ind]
    print(image.shape)
    coord = coords[ind]

    toast_filter(image,coord)