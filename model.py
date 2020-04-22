import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.io import imshow

class Model:

    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
        self.model = self.get_model()

    # Define the architecture
    def get_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=(self.max_y, self.max_x, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(64, kernel_size=1, strides=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Flatten())
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(18))
        return model

    def compile_model(self):
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['accuracy'])

    def train_model(self, imgs_train, points_train):
        checkpoint = ModelCheckpoint(filepath='weights/checkpoint-{epoch:02d}.hdf5')
        self.model.fit(imgs_train, points_train, epochs=300, batch_size=30, callbacks=[checkpoint])
    
    # Load weights for a previously trained model
    def load_trained_model(self):
        self.model.load_weights('weights/checkpoint-300.hdf5')

    # Function which plots an image with it's corresponding keypoints
    def visualize_points(self, img, coords):
        # fig,ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # imshow(img)
        # for i in range(0,len(points),2):
        #     x_renorm = (points[i]+0.5)*96       # Denormalize x-coordinate
        #     y_renorm = (points[i+1]+0.5)*96     # Denormalize y-coordinate
        #     circ = Circle((x_renorm, y_renorm),1, color='r')    # Plot the keypoints at the x and y coordinates
        #     ax.add_patch(circ)
        # plt.show()
        plt.imshow(img)
        plt.scatter(coords[:,0],coords[:,1], c='k')
        plt.show()

    # Testing the model
    def test_model(self, imgs_test):    
        # data_path = join('','*g')
        # files = glob.glob('./cat-dataset/CAT_01/00000100_002.jpg')
        # for i,f1 in enumerate(files):       # Test model performance on a screenshot for the webcam
        #     if f1 == 'Capture.PNG':
        #         img = imread(f1)
        #         img = rgb2gray(img)         # Convert RGB image to grayscale
        #         test_img = resize(img, (96,96))     # Resize to an array of size 96x96
        # test_img = np.array(test_img)
        # test_img_input = np.reshape(test_img, (1,96,96,1))      # Model takes input of shape = [batch_size, height, width, no. of channels]
        # prediction = model.predict(test_img_input)      # shape = [batch_size, values]
        # visualize_points(test_img, prediction[0])
        
        # Test on first 10 samples of the test set
        for i in range(len(imgs_test)):
            # test_img_input = np.reshape(imgs_test[i], (1,96,96,1))      # Model takes input of shape = [batch_size, height, width, no. of channels]
            prediction = self.model.predict(imgs_test[i])      # shape = [batch_size, values]
            print(prediction)
            self.visualize_points(imgs_test[i], prediction[0])
            if i == 10:
                break
