import os

import numpy as np

# params
DATA_NAME_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
RAW_LABELS = {
  'Hold': 0,
  'Rest': 1,
  'Preparation': 2,
  'Retraction': 3,
  'Stroke': 4
}
PRO_LABELS = {
    'H': 0,
    'D': 1,
    'P': 2,
    'R': 3,
    'S': 4
}


WINDOW_SIZE = 8
SPLIT_SIZE = 0.3  # 30%


def split_data(data_name_list, given_labels, data_type, window_size, split_size, test_label_path):
    test_label_file = open(test_label_path, 'w')
    test_label_file.write('index, label\n')

    num_total_test_data = 0
    for data_name in data_name_list:
        data = []

        raw_data_path = os.path.join('dataset/original/', '%s_%s.csv' % (data_name, data_type))
        with open(raw_data_path, 'r') as f:
            f.readline()
            for row in f:
                data.append(row.rstrip().split(','))

        data_array = np.array(data)
        labels = data_array[:, -1]
        # print(labels)
        print('label length = ', len(labels))
        # remove no-label data
        for i in range(len(labels)):
            if labels[i] not in given_labels:
                data_array = np.delete(data_array, i, 0)

        # make sliding window array data
        window_data_array = []
        for i in range(0, len(data_array) - window_size, 1):
            window_data_array.append(data_array[i: i + window_size, :])
        window_data_array = np.array(window_data_array)

        # data split
        # should NOT shuffle the data!
        # np.random.seed(123)
        # np.random.shuffle(window_data_array)
        num_train_data = int(len(window_data_array) * (1.0 - split_size))
        # print('num of train data =', num_train_data)

        if data_type == 'raw':
            train_data = window_data_array[:num_train_data, :, :-2]
            print(np.shape(train_data))  #: (x, 8, 18)
            train_label = window_data_array[:num_train_data, :, -1]
            # train_label = train_label.reshape((-1, WINDOW_SIZE, 1))
            print(np.shape(train_label))  #: (x, 8, 1)

            test_data = window_data_array[num_train_data:, :, :-2]
            print(np.shape(test_data))  # : (x, 8, 18)
            test_label = window_data_array[num_train_data:, :, -1]
            # test_label = test_label.reshape((-1, WINDOW_SIZE, 1))
            print(np.shape(test_label))  # : (x, 8, 1)
        else:
            train_vel_data = window_data_array[:num_train_data, :, :12]
            print(np.shape(train_vel_data))  #: (x, 8, 12)
            train_vel_label = window_data_array[:num_train_data, :, -1]
            # train_vel_label = train_vel_label.reshape((-1, WINDOW_SIZE, 1))
            print(np.shape(train_vel_label))
            train_acc_data = window_data_array[:num_train_data, :, 12:24]
            train_acc_label = window_data_array[:num_train_data, :, -1]
            # train_acc_label = train_acc_label.reshape((-1, WINDOW_SIZE, 1))
            train_sca_data = window_data_array[:num_train_data, :, 24:32]
            train_sca_label = window_data_array[:num_train_data, :, -1]
            # train_sca_label = train_sca_label.reshape((-1, WINDOW_SIZE, 1))

            test_vel_data = window_data_array[num_train_data:, :, :12]
            test_vel_label = window_data_array[num_train_data:, :, -1]
            # test_vel_label = test_vel_label.reshape((-1, WINDOW_SIZE, 1))
            test_acc_data = window_data_array[num_train_data:, :, 12:24]
            test_acc_label = window_data_array[num_train_data:, :, -1]
            # test_acc_label = test_acc_label.reshape((-1, WINDOW_SIZE, 1))
            test_sca_data = window_data_array[num_train_data:, :, 24:32]
            test_sca_label = window_data_array[num_train_data:, :, -1]
            # test_sca_label = test_sca_label.reshape((-1, WINDOW_SIZE, 1))

        if data_type == 'raw':
            write_data(train_data, train_label, data_name, data_type, 'train', None)
            write_data(test_data, test_label, data_name, data_type, 'test', None)
        else:
            write_data(train_vel_data, train_vel_label, data_name, data_type, 'train', 'vel')
            write_data(train_acc_data, train_acc_label, data_name, data_type, 'train', 'acc')
            write_data(train_sca_data, train_sca_label, data_name, data_type, 'train', 'sca')

            write_data(test_vel_data, test_vel_label, data_name, data_type, 'test', 'vel')
            write_data(test_acc_data, test_acc_label, data_name, data_type, 'test', 'acc')
            write_data(test_sca_data, test_sca_label, data_name, data_type, 'test', 'sca')

        if data_type == 'raw':
            num_test_data = 0
            for line in test_label:
                for value in line:
                    test_label_file.write('%d, %d\n' % (num_total_test_data + 1, given_labels[value]))
                    num_test_data += 1
                    num_total_test_data += 1
            print('%s %s: %d test data' % (data_name, data_type, num_test_data))
        else:
            num_test_data = 0
            for line in test_vel_label:
                for value in line:
                    test_label_file.write('%d, %d\n' % (num_total_test_data + 1, given_labels[value]))
                    num_test_data += 1
                    num_total_test_data += 1
            print('%s %s: %d test data' % (data_name, data_type, num_test_data))

    test_label_file.close()
    print('total number of test data = %d' % num_total_test_data)


def write_data(data, label, data_name, data_type, index, sub_type):
    train_test_path = 'dataset/train_test/'
    if not os.path.exists(train_test_path):
        os.makedirs(train_test_path)       
    # write data
    if sub_type is None:
        data_path = os.path.join(train_test_path, '%s_%s_%s.csv' % (data_name, data_type, index))
    else:
        data_path = os.path.join(train_test_path, '%s_%s_%s_%s.csv' % (data_name, data_type, sub_type, index))
    with open(data_path, 'w') as f:
        for line in data:
            for value in line:
                f.write(','.join([str(x) for x in value]))
                f.write('\n')
    # write label
    if sub_type is None:
        label_path = os.path.join(train_test_path, '%s_%s_%s_label.txt' % (data_name, data_type, index))
    else:
        label_path = os.path.join(train_test_path, '%s_%s_%s_%s_label.txt' % (data_name, data_type, sub_type, index))
    with open(label_path, 'w') as f:
        # to make prediction for the last WINDOW_SIZE!
        for line in label:
            f.write(line[-1])
            f.write('\n')
            '''
            # below gives the same label with the data!
            for value in line:
                f.write(value[:])
                f.write('\n')
            '''

raw_test_label_path = os.path.join('dataset', 'raw_test_list.csv')
pro_test_label_path = os.path.join('dataset', 'va3_test_list.csv')

split_data(DATA_NAME_LIST, RAW_LABELS, 'raw', WINDOW_SIZE, SPLIT_SIZE, raw_test_label_path)
split_data(DATA_NAME_LIST, PRO_LABELS, 'va3', WINDOW_SIZE, SPLIT_SIZE, pro_test_label_path)
