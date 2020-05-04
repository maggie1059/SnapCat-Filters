import numpy as np
import os
import cv2
from cv2 import imread
from imutils import rotate_bound
#from convenience import rotate_bound
import argparse
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
from model import Model
from preprocess import get_cats,load_data

#TODO: LOAD IN TRAINED MODEL TO GET PREDICTED KEYPOINTS

def ashhat_filter(image, coord, output_image_path="cat_ashhat.png"):

    #remove first element (in the care of size 19)
    #coord = coord[1:]
    #reshape if necessary
    coord = coord.reshape(9,2)
    #use tips when points are out of bounds
    left_tip = coord[4,:]
    right_tip = coord[-2,:]

    #otherwise use base of ear
    left_out_ear = coord[3,:]
    right_out_ear = coord[-1,:]

    hat = cv2.imread('filters/ashhat.png',cv2.IMREAD_UNCHANGED)

    #mirror ashhat if left point is higher than right
    if left_tip[0] > right_tip[0]:
        hat = cv2.flip(hat,1)

    #set hat angle using tips only if out of bounds
    #set width accordingly
    if left_tip[1] <0 or left_tip[0] <0:
        angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
        hat_width = np.linalg.norm(right_tip - left_tip)
    else:
        angle = np.arctan((left_out_ear[1]-right_out_ear[1])/(left_out_ear[0]-right_out_ear[0]))
        hat_width = np.linalg.norm(right_out_ear - left_out_ear)

    rotate_hat = rotate_bound(hat,angle*180/np.pi)

    old_height, old_width, channel = hat.shape

    hat_height = old_height*hat_width/old_width
    hat = cv2.resize(rotate_hat, (int(hat_width),int(hat_height)))
    hw,hh,hc = hat.shape

    hat = np.array(hat,dtype = np.float64)
    hat /= 255.0

    new_chan = np.copy(hat[:,:,0])
    hat[:,:,0] = hat[:,:,2]
    hat[:,:,2] = new_chan

    image = np.array(image,dtype =np.float64)
    image /= 255.0

    new_image_chan = np.copy(image[:,:,0])
    image[:,:,0] = image[:,:,2]
    image[:,:,2] = new_image_chan

    for i in range(0,hw):       # Overlay the filter based on the alpha channel
        for j in range(0,hh):
            #use ear tips if out of bounds
            if left_tip[1] <0 or left_tip[0] <0:
                if hat[i,j,-1] != 0 and i+left_tip[1] >=0 and j+left_tip[0] >=0:
                    image[i+left_tip[1],j+left_tip[0],:] = hat[i,j,:-1]
            #use ear base otherwise
            else:
                if hat[i,j,-1] != 0 and i+left_out_ear[1]-hw >=0 and j+left_out_ear[0] >=0:
                    image[i+left_out_ear[1]-hw,j+left_out_ear[0],:] = hat[i,j,:-1]

    plt.imshow(image)
    #plt.show()
    plt.savefig(output_image_path)
    #cv2.imwrite("output.png", image * 255)
    return image


def james_filter(image,coord, output_image_path="cat_james.png"):

    #remove first element (in the care of size 19)
    # coord = coord[1:]
    #reshape if necessary
    coord = coord.reshape(9,2)
    #use tips when points are out of bounds
    left_tip = coord[4,:]
    right_tip = coord[-2,:]

    #otherwise use base of ear
    left_out_ear = coord[3,:]
    right_out_ear = coord[-1,:]

    #try using mouth point for face
    mouth_point = coord[2,:]


    james = cv2.imread('filters/james.png',cv2.IMREAD_UNCHANGED)

    #mirror james if left point is higher than right
    if left_out_ear[0] > right_out_ear[0]:
        james = cv2.flip(james,1)

    #set hat angle using tips only if out of bounds
    #set width accordingly
    '''
    if left_tip[1] <0 or left_tip[0] <0:
        angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
        hat_width = np.linalg.norm(right_tip - left_tip)
    else:
        angle = np.arctan((left_out_ear[1]-right_out_ear[1])/(left_out_ear[0]-right_out_ear[0]))
        hat_width = np.linalg.norm(right_out_ear - left_out_ear)
    '''
    angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
    james_width = np.linalg.norm(right_tip - left_tip)
    james_height = mouth_point[1] - min(right_tip[1],left_tip[1])
    #print("left_tip", left_tip)
    #print("right_tip", right_tip)
    #print("mouth_point", mouth_point)

    #print("height", james_height)
    #print("width",james_width)

    rotate_james = rotate_bound(james,angle*180/np.pi)

    old_height, old_width, channel = james.shape

    #hat_height = old_height*hat_width/old_width
    james = cv2.resize(rotate_james, (int(1.3*james_width),int(1.3*james_height)))
    hw,hh,hc = james.shape

    james = np.array(james,dtype = np.float64)
    james /= 255.0

    new_chan = np.copy(james[:,:,0])
    james[:,:,0] = james[:,:,2]
    james[:,:,2] = new_chan

    image = np.array(image,dtype =np.float64)
    image /= 255.0

    new_image_chan = np.copy(image[:,:,0])
    image[:,:,0] = image[:,:,2]
    image[:,:,2] = new_image_chan

    for i in range(0,hw):       # Overlay the filter based on the alpha channel
        for j in range(0,hh):
            if james[i,j,-1] != 0 and i+left_tip[1] >=0 and j+left_tip[0] >=0:
                image[i+left_tip[1],j+left_tip[0],:] = james[i,j,:-1]
            '''
            #use ear tips if out of bounds
            if left_tip[1] <0 or left_tip[0] <0:
                if hat[i,j,-1] != 0 and i+left_tip[1] >=0 and j+left_tip[0] >=0:
                    image[i+left_tip[1],j+left_tip[0],:] = hat[i,j,:-1]
            #use ear base otherwise
            else:
                if hat[i,j,-1] != 0 and i+left_out_ear[1]-hw >=0 and j+left_out_ear[0] >=0:
                    image[i+left_out_ear[1],j+left_out_ear[0],:] = hat[i,j,:-1]
            '''
    return image
    plt.imshow(image)
    # plt.show()
    plt.savefig(output_image_path)
    plt.close()



data_dir = os.path.dirname(__file__) + './cat-dataset/small_dataset'# + str(i)
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

    ashhat_filter(image,coord)
    james_filter(image,coord)
    '''
    coord = coord[1:]
    coord = coord.reshape(9,2)
    plt.imshow(image)
    plt.scatter(coord[2,0],coord[2,1], c='k')
    #plt.scatter(coord[-2,0],coord[-2,1], c='k')
    plt.show()
    '''


'''
    coord = coord[1:]
    coord = coord.reshape(9,2)


    plt.imshow(image)
    plt.scatter(coord[4,0],coord[4,1], c='k')
    plt.scatter(coord[-2,0],coord[-2,1], c='k')
    plt.show()

    #use 3 for left ear and -1 for right ear for ash-hat

    left_tip = coord[4,:]
    right_tip = coord[-2,:]

    left_out_ear = coord[3,:]
    right_out_ear = coord[-1,:]

    #hat = cv2.imread('filters/ashhat.png',-1)
    hat = cv2.imread('filters/ashhat.png',cv2.IMREAD_UNCHANGED)


    #set hat angle using tips only if out of bounds
    #set width accordingly
    if left_tip[1] <0 or left_tip[0] <0:
        angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
        hat_width = np.linalg.norm(right_tip - left_tip)
    else:
        angle = np.arctan((left_out_ear[1]-right_out_ear[1])/(left_out_ear[0]-right_out_ear[0]))
        hat_width = np.linalg.norm(right_out_ear - left_out_ear)
    #angle = np.arctan((left_out_ear[1]-right_out_ear[1])/(left_out_ear[0]-right_out_ear[0]))
    #angle = np.arctan((left_tip[1]-right_tip[1])/(left_tip[0]-right_tip[0]))
    rotate_hat = rotate_bound(hat,angle*180/np.pi)

    old_height, old_width, channel = hat.shape


    #set hat shape
    #hat_width = np.linalg.norm(right_out_ear - left_out_ear)
    #hat_width = np.linalg.norm(right_tip - left_tip)
    hat_height = old_height*hat_width/old_width
    hat = cv2.resize(rotate_hat, (int(hat_width),int(hat_height)))
    hw,hh,hc = hat.shape


    #hat = cv2.resize(rotate_hat, (image.shape[1],image.shape[0]))
    hat = np.array(hat,dtype = np.float64)
    hat /= 255.0
    image = np.array(image,dtype =np.float64)
    image /= 255.0
    #print(hat.shape)
    #print(np.max(hat))
    #print(np.mean(hat))
    #print(np.max(image))
    #print(np.mean(image))

    plt.imshow(hat[:,:,-1])
    plt.scatter(coord[4,0],coord[4,1], c='k')
    plt.scatter(coord[-2,0],coord[-2,1], c='k')
    plt.show()
    #print(hat[:,:,-1])

    #added_image = cv2.addWeighted(np.array(image),0.4,hat,0.1,0)

    #plt.imshow(added_image)
    #plt.scatter(coord[3,0],coord[3,1], c='k')
    #plt.scatter(coord[-1,0],coord[-1,1], c='k')
    #plt.show()

    #cv2.imwrite('combined.png', added_image)
    #print("left coords", left_tip)
    #print("right coords", right_tip)
    #count = 0
    for i in range(0,hw):       # Overlay the filter based on the alpha channel
        for j in range(0,hh):
            #use ear tips if out of bounds
            if left_tip[1] <0 or left_tip[0] <0:
                if hat[i,j,-1] != 0 and i+left_tip[1] >=0 and j+left_tip[0] >=0:
                    image[i+left_tip[1],j+left_tip[0],:] = hat[i,j,:-1]
            #use ear base otherwise
            else:
                if hat[i,j,-1] != 0 and i+left_out_ear[1]-hw >=0 and j+left_out_ear[0] >=0:
                    image[i+left_out_ear[1]-hw,j+left_out_ear[0],:] = hat[i,j,:-1]


    plt.imshow(image)
    #plt.scatter(coord[4,0],coord[4,1], c='k')
    #plt.scatter(coord[-2,0],coord[-2,1], c='k')
    #plt.scatter(start_point[1],start_point[0], c='k')
    plt.show()
    '''
