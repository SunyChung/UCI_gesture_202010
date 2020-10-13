import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling1D


PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_1d_data(data_name, data_type, coord_len=None, sub_type=None):
    train_test_path = 'dataset/train_test/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    line_length = 0
    for prefix in PREFIX_LIST:
        if sub_type is None:
            data = open(train_test_path + str(prefix) + '_' + str(data_name)
                        + '_' + str(data_type) + '.csv', 'r')
            label = open(train_test_path + str(prefix) + '_' + str(data_name)
                         + '_' + str(data_type) + '_label.txt', 'r')
        else:
            data = open(train_test_path + str(prefix) + '_' + str(data_name)
                        + '_' + str(sub_type) + '_' + str(data_type) + '.csv', 'r')
            label = open(train_test_path + str(prefix) + '_' + str(data_name)
                         + '_' + str(sub_type) + '_' + str(data_type) + '_label.txt', 'r')

        for line in data:
            line = line.rstrip().split(',')
            line_length = len(line)
            temp = []
            for value in line:
                temp = temp + [float(value)]
            # print(np.shape(temp))
            x.append(temp)

        for l in label:
            y.append(data_dict[l.rstrip()])

    x = np.array(x).reshape((-1, line_length, WINDOW_SIZE))
    y = np.array(y).reshape((-1, 1, WINDOW_SIZE))
    print(np.shape(x))
    print(np.shape(y))
    return x, y, line_length

#raw_train_x, raw_train_y = load_1d_data('raw', 'train')
#raw_test_x, raw_test_y = load_1d_data('raw', 'test')

def load_T_1d_data(data_name, data_type, coord_length, sub_type=None):
    train_test_path = 'dataset/train_test/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    for prefix in PREFIX_LIST:
        if sub_type is None:
            data = open(train_test_path + str(prefix) + '_' + str(data_name)
                        + '_' + str(data_type) + '.csv', 'r')
            label = open(train_test_path + str(prefix) + '_' + str(data_name)
                         + '_' + str(data_type) + '_label.txt', 'r')
        else:
            data = open(train_test_path + str(prefix) + '_' + str(data_name)
                        + '_' + str(sub_type) + '_' + str(data_type) + '.csv', 'r')
            label = open(train_test_path + str(prefix) + '_' + str(data_name)
                         + '_' + str(sub_type) + '_' + str(data_type) + '_label.txt', 'r')

    line_length = 0
    for line in data:
        line = line.rstrip().split(',')
        temp = []
        for value in line:
            temp = temp + [float(value)]
        temp = np.array(temp)
        x.append(temp)
    print(np.shape(x))
    row_1, row_2, row_3 = [], [], []
    for line in x:
        line_length = len(line)
        for i in range((line_length // coord_length)):
            row_1.append(line[0 + (coord_length * i)])
            row_2.append(line[1 + (coord_length * i)])
            row_3.append(line[2 + (coord_length * i)])
    x_T = np.stack((row_1, row_2, row_3))
    x_T = np.array(x_T).reshape((-1, coord_length, (line_length//coord_length), WINDOW_SIZE))
    print(np.shape(x_T))

    for l in label:
        y.append(data_dict[l.rstrip()])
    print(np.shape(y))
    # check the array shape!!
    y = np.array(y).reshape((-1, 1, WINDOW_SIZE))
    print(np.shape(y))
    return x_T, y, line_length


#raw_train_x, raw_train_y, line_length = load_T_1d_data('raw', 'train', coord_length=3)


'''
raw_train_x, raw_train_y = load_1d_data('raw', 'train', coord_length=3)
raw_test_x, raw_test_y = load_1d_data('raw', 'test', coord_length=3)

va3_vel_train_x, va3_vel_train_y, row_length = load_1d_data('va3', 'train', coord_length=3, sub_type='vel')
va3_acc_train_x, va3_acc_train_y, row_length = load_1d_data('va3', 'train', coord_length=3, sub_type='acc')
va3_sca_train_x, va3_sca_train_y, row_length = load_1d_data('va3', 'train', coord_length=2, sub_type='sca')

va3_vel_test_x, va3_vel_test_y, row_length = load_1d_data('va3', 'test', coord_length=3, sub_type='vel')
va3_acc_test_x, va3_acc_test_y, row_length = load_1d_data('va3', 'test', coord_length=3, sub_type='acc')
va3_sca_test_x, va3_sca_test_y, row_length = load_1d_data('va3', 'test', coord_length=2, sub_type='sca')
'''


def build_model(filter_size, kernel_size, pool_size, coord_length, line_length):
    model = Sequential()
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size,
                     strides=1, activation='relu',
                     input_shape=(coord_length, (line_length//coord_length), WINDOW_SIZE)))
    model.add(Flatten())
    model.add(Dense(96, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(filter_size, kernel_size, pool_size, coord_length, num_epochs, batch_size, model_name):
    raw_train_x, raw_train_y, line_length = load_T_1d_data('raw', 'train', coord_length=3)
    raw_test_x, raw_test_y, line_length = load_T_1d_data('raw', 'test', coord_length=3)
    model = build_model(filter_size, kernel_size, pool_size, coord_length, line_length)

    for epoch in range(num_epochs):
        history = model.fit(raw_train_x, raw_train_y, validation_split=0, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True)
        _, accuracy = model.evaluate(raw_test_x, raw_test_y, batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))
    model.save_weights('./model/classifier_%s_weights.hdf5' % model_name)
    model.save('./model/classifier_%s.h5' % model_name)


def load_best(filter_size, kernel_size, pool_size, coord_length, batch_size, model_name):
    raw_test_x, raw_test_y, line_length = load_T_1d_data('raw', 'test', coord_length=3)

    model = build_model(filter_size, kernel_size, pool_size, coord_length, line_length)
    model.load_weights('./model/classifier_%s_weights.hdf5' % model_name)
    _, accuracy = model.evaluate(raw_test_x, raw_test_y, batch_size=batch_size, verbose=1)
    print('evaluation : accuracy(%) = ', round(accuracy, 3) * 100)


def main():
    coord_length = 3
    filepath = './model/'
    if not os.path.exists:
        os.makedirs(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    filter_size = 32
    kernel_size = 6 # or 3 ??
    pool_size = 3
    batch_size = 1
    num_epochs = 200
    model_name = '1d_raw_coord_3'

    run_model(filter_size, kernel_size, pool_size, coord_length, num_epochs, batch_size, model_name)
    #load_best(filter_size, kernel_size, pool_size, coord_len, num_epochs, batch_size, model_name)


if __name__ == '__main__':
    main()
