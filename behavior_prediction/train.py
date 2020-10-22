import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

from behavior_prediction.data_loader import *

WINDOW_SIZE = 8

# evaluation: accuracy(%)=   35.199999999999996
def build_model_dense():
    model = Sequential()
    model.add(Dense(512, input_shape=(1, 18), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_1D(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=1, strides=1, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_2D(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=1, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_3D(input_shape):
    model = Sequential()
    model.add(Conv3D(filters=2, kernel_size=1, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint):
    # train_x, train_y = one_to_one_data_load(data_type='raw', index='train', return_type='1D')
    # test_x, test_y = one_to_one_data_load(data_type='raw', index='test', return_type='1D')

    # train_x, train_y = per_win_data_load(data_type='raw', index='train', return_type='1D')
    # test_x, test_y = per_win_data_load(data_type='raw', index='test', return_type='1D')

    # train_x, train_y = per_win_data_load(data_type='raw', index='train', return_type='2D')
    # test_x, test_y = per_win_data_load(data_type='raw', index='train', return_type='2D')

    train_x, train_y = per_win_data_load(data_type='raw', index='train', return_type='3D')
    test_x, test_y = per_win_data_load(data_type='raw', index='train', return_type='3D')

    for epoch in range(num_epochs):
        history = model.fit(x=train_x, y=train_y, validation_split=0.1, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))

    model.save_weights('%s_weights.hdf5' %model_name)
    model.save('%s.h5' %model_name)


def load_best(data_name, model, batch_size, model_name):
    # test_x, test_y = per_win_data_load(data_type='raw', index='test', return_type='1D')
    # test_x, test_y = per_win_data_load(data_type='raw', index='test', return_type='2D')
    test_x, test_y = per_win_data_load(data_type='raw', index='train', return_type='3D')

    model.load_weights('%s_weights.hdf5' % model_name)
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation: accuracy(%)=  ', round(accuracy, 3)*100)


def main():
    file_path = './model/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 200

    # model_name = file_path + 'raw_1D_linear'
    # model_name = file_path + 'raw_1D_test'
    # model_name = file_path + 'raw_2D_test'
    model_name = file_path + 'raw_3D_test'
    data_name = 'raw'

    # model = build_model()
    # model = build_model_dense()
    # input_shape_1D = (WINDOW_SIZE, 18)  # 1D input shape
    # input_shape_2D = (WINDOW_SIZE, 18, 1)  # 2D input shape
    input_shape_3D = (1, WINDOW_SIZE, 18, 1)  # 3D input shape

    # model = build_model_1D(input_shape_1D)
    # model = build_model_2D(input_shape_2D)
    model = build_model_3D(input_shape_3D)

    run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint)
    load_best(data_name, model, batch_size, model_name)

if __name__ == '__main__':
    main()