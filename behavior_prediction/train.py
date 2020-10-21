import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

from behavior_prediction.data_loader import *

WINDOW_SIZE = 8


def build_model(n_latent):
    model = Sequential()
    model.add(Dense(512, input_shape=(WINDOW_SIZE, 18), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_latent))
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


def main():
    file_path = './model/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    batch_size = 16
    num_epochs = 10

    model_name = file_path + 'raw_1D_linear'
    data_name = 'raw'

    model = build_model(n_latent=5)
    run_model(data_name, model, num_epochs, batch_size, model_name, checkpoint)


if __name__ == '__main__':
    main()