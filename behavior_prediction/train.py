import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from behavior_prediction.data_loader import *

WINDOW_SIZE = 8

def build_2018():  # why is it still 50 %??
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 3), strides=(1, 3), activation='relu', input_shape=(8, 18, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model


def build_LSTM():
    model = Sequential()
    model.add(LSTM(16, input_shape=(8, 18)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


#def build_combined():



def run_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, test_x, test_y):
    for epoch in range(num_epochs):
        history = model.fit(x=train_x, y=train_y, validation_split=0.1, epochs=1,
                            batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        #  print(history.history.keys())
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))
    model.save_weights('%s_weights.hdf5' %model_name)
    model.save('%s.h5' %model_name)
    return history



def load_best(model, batch_size, model_name, test_x, test_y):
    model.load_weights('%s_weights.hdf5' % model_name)
    # model.load_weights('./model/checkpoint.hdf5')
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation: accuracy(%)=  ', round(accuracy, 3)*100)


def main():
    file_path = './model/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    checkpoint = ModelCheckpoint(filepath=file_path + 'checkpoint.hdf5', monitor='val_accuracy',
                                 verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 200

    # model_name = file_path + 'previous'  # 2018 model : 49.3 %
    model_name = file_path + 'raw_LSTM'  # 46.50

    # model = build_2018()
    model = build_LSTM()

    # return_type = '2D'
    return_type = 'LSTM'
    train_x, train_y = data_load(data_type='raw', index='train', return_type=return_type)
    print(np.shape(train_x[0]))  # (8, 18, 1)
    print(np.shape(train_y[0]))  # (1,)
    test_x, test_y = data_load(data_type='raw', index='test', return_type=return_type)

    history = run_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, test_x, test_y)
    load_best(model, batch_size, model_name, test_x, test_y)


    def plot_history(history):
        print(history.history)
        plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
        plt.title('training accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()


if __name__ == '__main__':
    main()