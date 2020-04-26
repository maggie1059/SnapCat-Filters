import tensorflow as tf
# import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model

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

        net = Dense(128, activation='relu') #(mobilenetv2_model.layers[-1].output)
        net = Dense(64, activation='relu') #(net)
        net = Dense(18, activation='linear') #(net)

        # model = Model(inputs=inputs, outputs=net)

        # model.summary()
        model = Sequential()
        model.add(mobilenetv2_model)
        model.add(net)
        # mobilenetv2_model = mobilenetv2.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling='max')
        # model.add(mobilenetv2_model)
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(18, activation='linear'))

        # model = Model(inputs=inputs, outputs=net)
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu, input_shape=(self.max_y, self.max_x, 1)))
        # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu))
        # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(256, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu))
        # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Conv2D(512, kernel_size=3, strides=2, padding='same', activation=tf.nn.leaky_relu))
        # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        # model.add(BatchNormalization())
        # model.add(Flatten())
        
        # model.add(Dense(512, activation=tf.nn.leaky_relu))
        # model.add(Dropout(0.1))
        # model.add(Dense(256, activation=tf.nn.leaky_relu))
        # model.add(Dropout(0.1))
        # model.add(Dense(128, activation=tf.nn.leaky_relu))
        # model.add(Dropout(0.1))
        # model.add(Dense(18))
        model.summary()
        return model

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), metrics = [self.accuracy])

    def get_imgs(self, folder):
        img_path = 'processed_train_imgs/img' + str(folder) + '.npy'
        coord_path = 'processed_train_coords/coord' + str(folder) + '.npy'
        images = np.load(img_path)
        coords = np.load(coord_path)
        for image in images:
            image = gray2rgb(image)
        print(images.shape)
        return images, coords
    
    def train_model(self):
        checkpoint = ModelCheckpoint(filepath='weights/checkpoint.hdf5', monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
        epochs = 150
        batch_size = 32

        for j in range(epochs):
            print("epoch: ", j)
            folder = j%7
            images, coords = self.get_imgs(folder)
            # randomly mirror images
            # images, coords, = preprocess.mirror(images, coords)
            images = np.expand_dims(images, axis=-1)
            coords = np.reshape(coords, (-1, 18))

            # shuffle examples and indices in same way
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            X = images[indices]
            Y = coords[indices]
            # for each batch
            self.model.fit(X, Y, validation_split=0.1, epochs=1, batch_size=batch_size, callbacks=[checkpoint, ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, mode="auto")])
            gc.collect()
    
    def get_test_data(self):
        test_img_path = "processed_test_imgs/img0.npy"
        test_coord_path = "processed_test_coords/coord0.npy"
        test_images = np.load(test_img_path)
        test_coords = np.load(test_coord_path)
        test_images = np.expand_dims(test_images, axis=-1)
        test_coords = np.reshape(test_coords, (-1, 18))
        return test_images, test_coords

    # Load weights for a previously trained model
    def load_trained_model(self):
        self.model.load_weights('weights/checkpoint.hdf5')

    # Function which plots an image with it's corresponding keypoints
    def visualize_points(self, img, coords, img_index):
        img_copy = np.copy(img)
        self.test(np.reshape(img, (1, 224, 224)), coords)
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

    def accuracy(self, y_true, y_pred, threshold=16):
        y_true = tf.reshape(y_true, (-1,9,2))
        y_pred = tf.reshape(y_pred, (-1,9,2))
        distances = tf.norm(y_pred - y_true, axis=2)
        out = tf.where(distances<threshold, 1., 0.)
        acc = tf.reduce_mean(out)
        # print(acc)
        return acc

    def test(self, test_imgs, test_coords):
        """ Testing routine. """
        test_imgs = np.expand_dims(test_imgs, axis=-1)
        test_coords = np.reshape(test_coords, (-1, 18))

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
