import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.io import imshow
from skimage.color import gray2rgb
import os
import gc
import preprocess

class Model():

    def __init__(self, max_x, max_y):
        super(Model, self).__init__() 
        self.max_x = max_x
        self.max_y = max_y
        self.model = self.get_model()

    # Define the architecture
    def get_model(self):
        inputs = Input(shape=(224, 224, 3))

        mobilenetv2_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')
        model = Sequential()
        model.add(mobilenetv2_model)
        
        model.add(Dense(512, activation=tf.nn.leaky_relu))
        model.add(Dropout(0.1))
        model.add(Dense(256, activation=tf.nn.leaky_relu))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation=tf.nn.leaky_relu))
        model.add(Dropout(0.1))
        model.add(Dense(18))
        model.summary()
        return model

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics = [self.accuracy])

    def get_imgs(self, folder):
        img_path = 'processed_train_imgs/img' + str(folder) + '.npy'
        coord_path = 'processed_train_coords/coord' + str(folder) + '.npy'
        images = np.load(img_path)
        coords = np.load(coord_path)
        new_ims = np.zeros((images.shape[0], 224, 224, 3))
        for i in range(len(images)):
            image = gray2rgb(images[i])
            new_ims[i] = image
        return new_ims, coords
    
    def train_model(self):
        checkpoint = ModelCheckpoint(filepath='weights/checkpoint.hdf5', monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
        epochs = 100
        batch_size = 32

        for j in range(epochs):
            print("epoch: ", j)
            folder = j%7
            images, coords = self.get_imgs(folder)
            coords = np.reshape(coords, (-1, 18))

            # shuffle examples and indices in same way
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            X = images[indices]
            Y = coords[indices]
            # for each batch
            self.model.fit(X, Y, validation_split=0.1, epochs=1, batch_size=batch_size, callbacks=[checkpoint])
            gc.collect()
    
    def get_test_data(self):
        test_img_path = "processed_test_imgs/img0.npy"
        test_coord_path = "processed_test_coords/coord0.npy"
        test_images = np.load(test_img_path)
        test_coords = np.load(test_coord_path)
        test_coords = np.reshape(test_coords, (-1, 18))
        return test_images, test_coords

    # Load weights for a previously trained model
    def load_trained_model(self):
        self.model.load_weights('weights/checkpoint.hdf5')

    # Function which plots an image with it's corresponding keypoints
    def visualize_points(self, img, coords, img_index):
        img_c = np.copy(img)
        img_copy = gray2rgb(img_c)
        self.test(np.reshape(img_copy, (1, 224, 224, 3)), coords)
        img = np.reshape(img_copy, (1, 224, 224, 3))
        predicted = self.model.predict(img, verbose=1)
        predicted = tf.reshape(predicted, (9,2))
        predicted = predicted.numpy()
        print("PREDICTED: ", predicted)
        plt.imshow(img_c)
        plt.scatter(coords[:,0],coords[:,1], c='k')
        plt.scatter(predicted[:,0],predicted[:,1], c='r')
        plt.show()
        plt.savefig("output" + str(img_index) + ".png")
        plt.close()

    def accuracy(self, y_true, y_pred, threshold=6):
        y_true = tf.reshape(y_true, (-1,9,2))
        y_pred = tf.reshape(y_pred, (-1,9,2))
        distances = tf.norm(y_pred - y_true, axis=2)
        out = tf.where(distances<threshold, 1., 0.)
        acc = tf.reduce_mean(out)
        return acc

    def test(self, test_imgs, test_coords):
        """ Testing routine. """
        new_ims = np.zeros((test_imgs.shape[0], 224, 224, 3))
        for i in range(len(test_imgs)):
            image = gray2rgb(test_imgs[i])
            new_ims[i] = image
        test_coords = np.reshape(test_coords, (-1, 18))

        # Run model on test set
        self.model.evaluate(
            x=new_ims,
            y=test_coords,
            verbose=1,
        )
