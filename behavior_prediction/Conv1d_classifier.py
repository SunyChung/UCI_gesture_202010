import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

prefix_list = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8

def load_data(label_type, data_type):
    train_test_path = 'dataset/train_test/'
    if label_type == 'raw':
        label_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        label_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3,'S': 4}
    x = []
    y = []
    for prefix in prefix_list:
        data = open(train_test_path + str(prefix) + '_' + str(label_type) + '_' + str(data_type) + '.csv', 'r')
        label = open(train_test_path + str(prefix) + '_' + str(label_type) + '_' + str(data_type) + '_label.txt', 'r')
        for d in data:
            temp = []
            for a in d.rstrip().split(','):
                temp = temp + [float(a)]
            x.append(temp)
        for l in label:
            temp = [0, 0, 0, 0, 0]
            temp[label_dict[l.rstrip()]] = 1
            y.append(temp)
    x = np.array(x).reshape((len(y), WINDOW_SIZE, 18))
    y = np.array(y).reshape((len(y), 5))
    return x, y

def get_all_data():
    train_x, train_y = load_data('raw', 'train')
    test_x, test_y = load_data('raw', 'test')
    return train_x, train_y, test_x, test_y

def build_model(filter_no):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 18)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam', #Adam(lr=0.0001, epsilon=1e-4),
              metrics=['accuracy'])
    return model


filepath = "./model/best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                            verbose=1, save_best_only=True, mode='max')
batch_size = 16
num_epochs = 200
k = 10

def parameter_test():
    for filter_no in [16, 32, 48, 64]:
        train_x, train_y, test_x, test_y = get_all_data()
        num_val_samples = len(train_x) // k
        accuracy_list = []
        for i in range(k):
            print('processing fold #: ', i)
            val_data = train_x[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_y[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                                                [train_x[:i * num_val_samples],
                                                train_x[(i + 1) * num_val_samples:]],
                                                axis=0)
            partial_train_targets = np.concatenate(
                                                    [train_y[:i * num_val_samples],
                                                    train_y[(i + 1) * num_val_samples:]],
                                                    axis=0)
            model = build_model(filter_no)
            model.fit(partial_train_data, partial_train_targets,
                      epochs=num_epochs, batch_size=16, verbose=0,
                      callbacks=[checkpoint])
            _, accuracy = model.evaluate(val_data, val_targets, batch_size=16)
            accuracy_list.append(accuracy)
        print('\naccuracy validation: filter number = ', filter_no)
        print(accuracy_list)
        print('\nvalidation score average = ', np.mean(accuracy_list))


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
    test_x, test_y = load_data('raw', 'test')
    model.load_weights('./classifier_weights.hdf5')
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    print('evaluation: accuracy(%)=  ', round(accuracy, 3)*100)


def main():
    #parameter_test()
    run_model(filter_no=32)
    #load_best(filter_no=64)


if __name__ == "__main__":
    main()