import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

# from behavior_prediction.data_loader import data_load
from behavior_prediction.load_split_data import featured_data_load, split_data_load
from behavior_prediction.models import build_2018, build_LSTM, build_concate, build_p_feature, build_concate

WINDOW_SIZE = 8


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


def run_feature_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, train_f, test_x, test_y, test_f):
    for epoch in range(num_epochs):
        history = model.fit({'coordinate': train_x, 'person': train_f},y=train_y,
                            validation_split=0, epochs=1, batch_size=batch_size,
                            verbose=1, shuffle=True, callbacks=[checkpoint]
                            )
        _, accuracy = model.evaluate({'coordinate': test_x, 'person': test_f}, y=test_y,
                                     batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))
    model.save_weights('%s_weights.hdf5' %model_name)
    model.save('%s.h5' %model_name)
    return history


def run_concatenated(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, train_f, test_x, test_y, test_f):
    for epoch in range(num_epochs):
        history = model.fit({'left_hand': train_x[0], 'right_hand': train_x[:, 1, :, :],
                             'head': train_x[:, 2, :, :], 'spine': train_x[:, 3, :, :],
                             'left_wrist': train_x[:, 4, :, :], 'right_wrist': train_x[:, 5, :, :],
                             'who': train_f}, y=train_y,
                            validation_split=0, epochs=1, batch_size=batch_size,
                            verbose=1, shuffle=True, callbacks=[checkpoint])
        _, accuracy = model.evaluate({'left_hand': test_x[:, 0, :, :], 'right_hand': test_x[:, 1, :, :],
                                      'head': test_x[:, 2, :, :], 'spine': test_x[:, 3, :, :],
                                      'left_wrist': test_x[:, 4, :, :], 'right_wrist': test_x[:, 5, :, :],
                                      'who': test_f}, y=test_y,
                                     batch_size=batch_size, verbose=1)
        print('%d : accuracy = %f' %(epoch, round(accuracy, 3) * 100))
    model.save_weights('%s_weights.hdf5' %model_name)
    model.save('%s.h5' %model_name)
    return history


def load_best(model, batch_size, model_name, test_x, test_y, test_f):
    model.load_weights('%s_weights.hdf5' % model_name)
    # model.load_weights('./model/checkpoint.hdf5')
    _, accuracy = model.evaluate({'coordinate': test_x, 'person': test_f}, y=test_y,
                                 batch_size=batch_size, verbose=1)
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
    # model_name = file_path + 'raw_LSTM'  # 46.50
    # model_name = file_path + 'p_features'  # 46.50
    model_name = file_path + 'concatenated'
    model = build_concate()

    train_x, train_y, train_f  = split_data_load(data_type='raw', index='train')
    print('data loaded')
    print(np.shape(train_x[0]))
    print(np.shape(train_y))
    test_x, test_y, test_f = split_data_load(data_type='raw', index='test')

    history = run_concatenated(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, train_f, test_x, test_y, test_f)
    load_best(model, batch_size, model_name, test_x, test_y, test_f)


    def plot_history(history):
        print(history.history)
        plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
        plt.title('training accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()


if __name__ == '__main__':
    main()