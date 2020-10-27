import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

# from behavior_prediction.data_loader import data_load
from behavior_prediction.load_split_data import featured_data_load
from behavior_prediction.models import build_2018, build_LSTM, build_concate, build_p_feature

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


def run_feature_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, train_f, test_x, test_y):
    for epoch in range(num_epochs):
        history = model.fit(
            {'coordinate': train_x,
             'label': train_y,
             'person': train_f
             },
            validation_split=0, epochs=1, batch_size=batch_size, verbose=1, shuffle=True, callbacks=[checkpoint]
        )
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
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
    # model_name = file_path + 'raw_LSTM'  # 46.50
    model_name = file_path + 'p_features'  # 46.50

    # model = build_2018()
    model = build_p_feature()

    # return_type = '2D'
    # return_type = 'LSTM'
    # train_x, train_y = data_load(data_type='raw', index='train', return_type=return_type)
    train_x, train_y, train_f  = featured_data_load(data_type='raw', index='train')
    print(np.shape(train_x[0]))  # (8, 18)
    print(np.shape(train_y[0]))  # (1,)
    print(np.shape(train_f[0]))  # (8, 1, 1)
    # test_x, test_y = data_load(data_type='raw', index='test', return_type=return_type)
    test_x, test_y, test_f = featured_data_load(data_type='raw', index='test')
    print(np.shape(test_x[0]))
    print(np.shape(test_y[0]))

    # history = run_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, test_x, test_y)
    history = run_feature_model(model, num_epochs, batch_size, model_name, checkpoint, train_x, train_y, train_f, test_x, test_y)
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