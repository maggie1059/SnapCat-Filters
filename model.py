import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

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
        self.model.fit(imgs_train, points_train, epochs=300, batch_size=100, callbacks=[checkpoint])

