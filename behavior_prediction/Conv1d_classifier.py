import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_1d_data(data_name, data_type, sub_type=None):
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

        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for value in line:
                temp = temp + [float(value)]
            # print(np.shape(temp))
            x.append(temp)

        for l in label:
            #print(l.rstrip().replace(',', ''))  # R,e,s,t -> Rest
            y.append(data_dict[l.rstrip().replace(',', '')])

    x = np.array(x).reshape((-1, 1, 18))
    y = np.array(y).reshape((-1, 1, 1))
    print(np.shape(x))
    print(np.shape(y))
    return x, y


#x, y = load_1d_data('raw', 'train', sub_type=None)
#print(x)
#print(y)


def get_all_data():
    train_x, train_y = load_1d_data('raw', 'train')
    test_x, test_y = load_1d_data('raw', 'test')
    return train_x, train_y, test_x, test_y


def build_model(filter_no):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(18, )))
    # input shape [16, 1, 1, 18]
    # conv1D [1, 3, 18, 32]
    # model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


filepath = "./model/best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
batch_size = 16
num_epochs = 200
k = 10


def run_model(filter_no):
    model = build_model(filter_no)
    train_x, train_y, test_x, test_y = get_all_data()
    for epoch in range(num_epochs):
        history = model.fit(train_x, train_y, validation_split=0, epochs=1, batch_size=batch_size, verbose=1, shuffle=True)
        _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
        print('%d: accuracy = %f' % (epoch, round(accuracy, 3)*100))
    model.save_weights('./classifier_weights.hdf5')
    model.save('./classifier.h5')


def load_best(filter_no):
    model = build_model(filter_no)
    test_x, test_y = load_1d_data('raw', 'test')
    model.load_weights('./classifier_weights.hdf5')
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation: accuracy(%)=  ', round(accuracy, 3)*100)


def main():
    run_model(filter_no=32)
    #load_best(filter_no=64)


if __name__ == "__main__":
    main()
