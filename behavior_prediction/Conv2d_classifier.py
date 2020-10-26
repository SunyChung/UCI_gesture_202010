import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint

from behavior_prediction.data_loader import *

WINDOW_SIZE = 8


def build_model(data_name):
    if data_name == 'raw':
        input_shape = (WINDOW_SIZE, 18, 1)
    else:
        input_shape = (WINDOW_SIZE, 32, 1)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_2(data_name):
    if data_name == 'raw':
        input_shape = (WINDOW_SIZE, 18, 1)
    else:
        input_shape = (WINDOW_SIZE, 32, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 1), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint):
    train_x, train_y = data_load(data_name, 'train', return_type='2D')
    test_x, test_y = data_load(data_name, 'test', return_type='2D')

    for epoch in range(num_epochs):
        history = model.fit(train_x, train_y, validation_split=0, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' % (epoch, round(accuracy, 3) * 100))

    model.save_weights('%s_weights.hdf5' % model_name)
    model.save('%s.h5' % model_name)


def load_best(data_name, model, batch_size, model_name):
    test_x, test_y = data_load(data_name, 'test', return_type='2D')

    model.load_weights('%s_weights.hdf5' % model_name)
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation : accuracy(%) = ', round(accuracy, 3) * 100)


def main():
    file_path = './model/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 200

    model_name = file_path + 'Conv2D'

    data_name = 'raw'

    model = build_model(data_name)
    run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint)
    load_best(data_name, model, batch_size, model_name)


if __name__ == '__main__':
    main()
