import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

from behavior_prediction.data_loader import *

WINDOW_SIZE = 8

# for data-label one-to-one data, Conv1D doesn't work
# : NN output shape != target label issue
def build_model(batch_size):
    model = Sequential()
    input_shape = (1, 18)
    model.add(Conv1D(filters=32, kernel_size=1, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(18, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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


def run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint):
    train_x, train_y = one_to_one_data_load(data_type='raw', index='train', return_type='1D')
    test_x, test_y = one_to_one_data_load(data_type='raw', index='test', return_type='1D')

    for epoch in range(num_epochs):
        history = model.fit(train_x, train_y, validation_split=0.1, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))

    model.save_weights('%s_weights.hdf5' %model_name)
    model.save('%s.h5' %model_name)


def load_best(data_name, model, batch_size, model_name):
    test_x, test_y = one_to_one_data_load(data_name, 'test', return_type='1D')
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
    model_name = file_path + 'raw_dense'
    data_name = 'raw'

    # model = build_model(batch_size)
    model = build_model_dense()
    run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint)
    load_best(data_name, model, batch_size, model_name)

if __name__ == '__main__':
    main()