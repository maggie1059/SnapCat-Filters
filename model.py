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
import os

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
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics = ["accuracy"])

    def get_imgs(self, folder):
        img_path = 'processed_train_imgs/img' + str(folder) + '.npy'
        coord_path = 'processed_train_coords/coord' + str(folder) + '.npy'
        images = np.load(img_path)
        images = np.expand_dims(images, axis=-1)
        coords = np.load(coord_path)
        coords = np.reshape(coords, (-1, 18,))
        return images, coords
    
    def train_model(self):
        checkpoint = ModelCheckpoint(filepath='weights/checkpoint.hdf5', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        epochs = 10
        batch_size = 10

        for j in range(epochs):
            print("epoch: ", j)
            folder = j%7
            images, coords = self.get_imgs(folder)
            # shuffle examples and indices in same way
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            X = images[indices]
            Y = coords[indices]
            # for each batch
            self.model.fit(X, Y, validation_split=0.2, epochs=1, batch_size=batch_size, callbacks=[checkpoint])
              
    
    # Load weights for a previously trained model
    def load_trained_model(self):
        self.model.load_weights('weights/checkpoint-699.hdf5')

    # Function which plots an image with it's corresponding keypoints
    def visualize_points(self, img, coords, img_index):
        img_copy = np.copy(img)
        img = np.reshape(img, (1, 224, 224, 1))
        predicted = self.model.predict(img, verbose=1)
        predicted = tf.reshape(predicted, (9,2))
        predicted = predicted.numpy()
        print("PREDICTED: ", predicted)
        plt.imshow(img_copy)
        plt.scatter(coords[:,0],coords[:,1], c='k')
        plt.scatter(predicted[:,0],predicted[:,1], c='r')
        plt.show()
        plt.savefig("output" + str(img_index) + ".png")
        plt.close()

    def accuracy(self, y_true, y_pred, threshold=3):
        y_true = tf.reshape(y_true, (-1,9,2))
        y_pred = tf.reshape(y_pred, (-1,9,2))
        distances = tf.norm(y_pred - y_true, axis=2)
        out = tf.where(distances<threshold, 1, 0)
        acc = tf.reduce_mean(out)
        # print(acc)
        return acc

    def test(self, test_imgs, test_coords):
        """ Testing routine. """
        test_imgs = np.expand_dims(test_imgs, axis=-1)
        test_coords = np.reshape(test_coords, (-1, 18,))

        # Run model on test set
        self.model.evaluate(
            x=test_imgs,
            y=test_coords,
            verbose=1,
        )

    # Testing the model
    # def test_model(self, imgs_test):    
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
        # for i in range(len(imgs_test)):
        #     # test_img_input = np.reshape(imgs_test[i], (1,96,96,1))      # Model takes input of shape = [batch_size, height, width, no. of channels]
        #     prediction = self.model.predict(imgs_test[i])      # shape = [batch_size, values]
        #     print(prediction)
        #     self.visualize_points(imgs_test[i], prediction[0])
        #     if i == 10:
        #         break
