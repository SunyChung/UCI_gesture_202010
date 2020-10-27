import numpy as np

PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
RAW_LABELS = {
  'Hold': 0,
  'Rest': 1,
  'Preparation': 2,
  'Retraction': 3,
  'Stroke': 4
}
WINDOW_SIZE = 8


def split_data_load(data_type, index):
    data_dir = 'dataset/multi_train_test/'
    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '.csv', 'r')
        label = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '_label.txt', 'r')
        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for i in range(0, len(line), 3):
                per_3 = [float(j) for j in line[i:i+3]]
                temp = temp + [per_3]
            x = np.concatenate(temp, axis=1)
        for value in label:
            y.append(int(value.rstrip()))

    print(np.shape(x))
    print(np.shape(y))
    #x = np.array(x).reshape((8, -1, 3))
    #y = np.array(y).reshape((-1, 1))
    return x, y

# data, label = split_data_load(data_type='raw', index='train')


def featured_data_load(data_type, index):
    data_dir = 'dataset/per_win_train_test/'
    x = []
    y = []
    l = []
    for prefix in PREFIX_LIST:
        data = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '.csv', 'r')
        label = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '_label.txt', 'r')
        # data array extraction
        for line in data:
            line = line.rstrip().split(',')
            if prefix[0] == 'a':
                l.append(int(0))  # person a -> 0
            elif prefix[0] == 'b':
                l.append(int(1))  # person b -> 1
            else:  # prefix[0] == 'c'
                l.append(int(2))  # person c -> 2
            temp = []
            for value in line:
                temp = temp + [float(value)]
            x.append(temp)
        # label data extraction
        for value in label:
            y.append(int(RAW_LABELS[value.rstrip()]))

    x = np.array(x).reshape((-1, WINDOW_SIZE, 18, 1))
    y = np.array(y).reshape((-1, 1))
    l = np.array(l).reshape((-1, WINDOW_SIZE, 1, 1))
    return x, y, l


data, label, feature = featured_data_load('raw', 'train')

