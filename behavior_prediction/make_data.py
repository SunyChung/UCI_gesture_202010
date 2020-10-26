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

WINDOW_SIZE = 8
SPLIT_SIZE = 0.3  # 30%

def make_data(data_name_list, given_labels, data_type, window_size, split_size):
    test_label_path = os.path.join('dataset', 'raw_test_list.csv')
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
        for i in range(len(labels)):
            if labels[i] not in given_labels:
                data_array = np.delete(data_array, i, 0)

        window_data_array = []
        for i in range(0, len(data_array) - window_size, 1):
            window_data_array.append(data_array[i: i + window_size, :])
        window_data_array = np.array(window_data_array)

        num_train_data = int(len(window_data_array) * (1.0 - split_size))
        print('num of train data =', num_train_data)

        train_data = window_data_array[:num_train_data, :, :-2]
        train_label = window_data_array[:num_train_data, :, -1]
        test_data = window_data_array[num_train_data:, :, :-2]
        test_label = window_data_array[num_train_data:, :, -1]

        write_data(train_data, train_label, data_name, data_type, index='train')
        write_data(test_data, test_label, data_name, data_type, index='test')


def write_data(data, label, data_name, data_type, index):
    data_dir = 'dataset/multi_train_test/'
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
        for line in label:
            # get the last label in the WINDOW
            # differentiate gesture label for each person
            # 15 label prediction problem
            last_label = line[-1].rstrip()
            if data_name[0] == 'b':
                f.write(str(RAW_LABELS[last_label] + 5))
            elif data_name[0] == 'c':
                f.write(str(RAW_LABELS[last_label] + 10))
            else:
                f.write(str(RAW_LABELS[last_label]))
            f.write('\n')


make_data(PREFIX_LIST, RAW_LABELS, 'raw', WINDOW_SIZE, SPLIT_SIZE)
