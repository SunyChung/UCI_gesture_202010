import os
import keras
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint


from behavior_prediction.data_loader import *

WINDOW_SIZE = 8


def get_all_data(data_name):
    train_x, train_y = data_load(data_name, 'train', return_type='1D')
    test_x, test_y = data_load(data_name, 'test', return_type='1D')
    return train_x, train_y, test_x, test_y


def build_model(data_name):
    if data_name == 'raw':
        input_shape = (WINDOW_SIZE, 18)
    else:
        input_shape = (WINDOW_SIZE, 32)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint):
    train_x, train_y, test_x, test_y = get_all_data(data_name)
    for epoch in range(num_epochs):
        history = model.fit(train_x, train_y, validation_split=0.1, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True,
                            callbacks=[checkpoint])
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print('%d: accuracy = %f' % (epoch, round(accuracy, 3)*100))
    model.save_weights('%s_weights.hdf5' % model_name)
    model.save('%s.h5' % model_name)


def load_best(data_name, model, batch_size, model_name):
    test_x, test_y = data_load(data_name, 'test', return_type='1D')
    model.load_weights('%s_weights.hdf5' % model_name)
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation: accuracy(%)=  ', round(accuracy, 3)*100)


def main():
    file_path = "./model/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 200

    model_name = file_path + 'Conv1D'

    data_name = 'raw'
    model = build_model(data_name)
    run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint)
    load_best(data_name, model, batch_size, model_name)


if __name__ == "__main__":
    main()
