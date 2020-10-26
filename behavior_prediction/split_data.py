import os
import numpy as np

# parameters
PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
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


def split_data(data_name_list, given_labels, data_type, window_size, split_size):
    if data_type == 'raw':
        test_label_path = os.path.join('dataset', 'raw_test_list.csv')
    else:
        test_label_path = os.path.join('dataset', 'va3_test_list.csv')
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
        # print('label length = ', len(labels))
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
        print('num of train data =', num_train_data)

        if data_type == 'raw':
            train_data = window_data_array[:num_train_data, :, :-2]
            print(np.shape(train_data))  #: (x, 8, 18)
            train_label = window_data_array[:num_train_data, :, -1]
            print(np.shape(train_label))  #: (x, 8, 1)

            test_data = window_data_array[num_train_data:, :, :-2]
            print(np.shape(test_data))  # : (x, 8, 18)
            test_label = window_data_array[num_train_data:, :, -1]
            print(np.shape(test_label))  # : (x, 8, 1)

            write_data(train_data, train_label, data_name, data_type, index='train', write_type='per_window')
            write_data(test_data, test_label, data_name, data_type, index='test', write_type='per_window')
            # write_data(train_data, train_label, data_name, data_type, index='train', write_type='one_to_one')
            # write_data(test_data, test_label, data_name, data_type, index='test', write_type='one_to_one')

        else:  # va3_data (not used)
            train_data = window_data_array[:num_train_data, :, :32]
            print(np.shape(train_data))  #: (x, 8, 32)
            train_label = window_data_array[:num_train_data, :, -1]
            print(np.shape(train_label))
            test_data = window_data_array[num_train_data:, :, :32]
            print(np.shape(test_data))
            test_label = window_data_array[num_train_data:, :, -1]
            print(np.shape(test_label))

            print('data-label : per window write\n')
            write_data(train_data, train_label, data_name, data_type, index='train', write_type='per_window')
            write_data(test_data, test_label, data_name, data_type, index='test', write_type='per_window')
            # print('data-label : one-to-one write')
            # write_data(train_data, train_label, data_name, data_type, index='train', write_type='one_to_one')
            # write_data(test_data, test_label, data_name, data_type, index='test', write_type='one_to_one')

        num_test_data = 0
        for line in test_label:
            num_test_data += 1
            num_total_test_data += 1
            for value in line:
                test_label_file.write('%d, %d\n' % (num_total_test_data + 1, given_labels[value]))
        print('%s %s: %d test data' % (data_name, data_type, num_test_data))
    test_label_file.close()
    print('total number of test data = %d' % num_total_test_data)


def write_data(data, label, data_name, data_type, index, write_type):
    if write_type == 'per_window':
        data_dir = 'dataset/per_win_train_test/'
    else:
        data_dir = 'dataset/one_to_one_train_test/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # write data
    data_path = os.path.join(data_dir, '%s_%s_%s.csv' % (data_name, data_type, index))
    with open(data_path, 'w') as f:
        for line in data:
            for value in line:
                f.write(','.join([str(x) for x in value]))
                f.write('\n')
    # write label
    label_path = os.path.join(data_dir, '%s_%s_%s_label.txt' % (data_name, data_type, index))
    with open(label_path, 'w') as f:
        # to make prediction for the last WINDOW_SIZE!
        if write_type == 'per_window':
            for line in label:
                f.write(line[-1])
                f.write('\n')
        else:  # data-label one-to-one corresponding write
            for line in label:
                for value in line:
                    f.write(value)
                    f.write('\n')


split_data(PREFIX_LIST, RAW_LABELS, 'raw', WINDOW_SIZE, SPLIT_SIZE)
# split_data(PREFIX_LIST, PRO_LABELS, 'va3', WINDOW_SIZE, SPLIT_SIZE)
