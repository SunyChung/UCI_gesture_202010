import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_1d_data(data_name, data_type):
    train_test_path = './dataset/train_test/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '.csv', 'r')
        label = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '_label.txt', 'r')

        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for value in line:
                temp = temp + [float(value)]
            # print(np.shape(temp))
            x.append(temp)

        for l in label:
            y.append(data_dict[l.rstrip()])
    # for Conv1D input!
    # x = np.array(x).reshape((-1, WINDOW_SIZE, 18))
    # for Conv2D input!!
    x = np.array(x).reshape((-1, WINDOW_SIZE, 18, 1))
    y = np.array(y).reshape((-1, 1))
    print(np.shape(x))
    print(np.shape(y))
    return x, y

'''
raw_train_x, raw_train_y = load_1d_data('raw', 'train')
raw_test_x, raw_test_y = load_1d_data('raw', 'test')

va3_train_x, va3_train_y = load_1d_data('va3', 'train')
va3_test_x, va3_test_y = load_1d_data('va3', 'test')
'''


def build_model():
    model = Sequential()
    # Conv2D : image size (width x height) X num of channel!
    # thus, the last dimension is 1 for this case!
    # SHOULD reshape the input data to (1, 8, 18, 1) for Conv2D!
    model.add(Conv2D(32, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=(WINDOW_SIZE, 18, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=(WINDOW_SIZE, 18, 1)))
    model.add(Conv2D(32, kernel_size=(3, 1), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(model, num_epochs, batch_size, model_name, checkpoint):
    raw_train_x, raw_train_y = load_1d_data('raw', 'train')
    raw_test_x, raw_test_y = load_1d_data('raw', 'test')

    for epoch in range(num_epochs):
        history = model.fit(raw_train_x, raw_train_y, validation_split=0, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate(raw_test_x, raw_test_y, batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' % (epoch, round(accuracy, 3) * 100))

    model.save_weights('%s_weights.hdf5' % model_name)
    model.save('%s.h5' % model_name)


def load_best(model, batch_size, model_name):
    raw_test_x, raw_test_y = load_1d_data('raw', 'test')

    model.load_weights('%s_weights.hdf5' % model_name)
    _, accuracy = model.evaluate(raw_test_x, raw_test_y, batch_size=batch_size, verbose=1)
    print('evaluation : accuracy(%) = ', round(accuracy, 3) * 100)


def main():
    file_path = './model/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # model_name = file_path + 'raw_Conv2D'
    model_name = file_path + 'raw_Conv2D_2'

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 200

    # model = build_model()
    model = build_model_2()
    run_model(model, num_epochs, batch_size, model_name, checkpoint)
    load_best(batch_size, model_name)


if __name__ == '__main__':
    main()
