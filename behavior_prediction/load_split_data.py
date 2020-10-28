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
    data_dir = 'dataset/per_win_train_test/'
    x = []
    y = []
    l = []
    for prefix in PREFIX_LIST:
        data = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '.csv', 'r')
        label = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '_label.txt', 'r')


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

        for value in label:
            y.append(int(RAW_LABELS[value.rstrip()]))

    print(x[0])
    # [5.347435, 4.363681, 1.501913,
    # 5.258967, 4.319263, 1.488703,
    # 5.037871, 1.618295, 1.77835,
    # 5.062803, 4.229656, 1.772577,
    # 4.972902, 4.301065, 1.564781,
    # 5.553945, 4.370456, 1.553521]
    print(x[1])
    # [4.869622, 4.25421, 1.556133,
    # 5.240113, 4.346338, 1.554309,
    # 5.03761, 1.61837, 1.778573,
    # 5.06143, 4.228504, 1.772859,
    # 4.974908, 4.303656, 1.565527,
    # 5.423875, 4.303708, 1.569942]
    x = np.split(np.array(x), 6, axis=1)  # (6, 55104, 3)
    # print(np.shape(x))  # (6, 55104, 3)

    # reshape() does NOT change the original data!!
    x = np.array(x).reshape((-1, 6, WINDOW_SIZE, 3))  # (6888, 6, 8, 3)
    y = np.array(y).reshape((-1, 1))  # (6888, 1)
    l = np.array(l).reshape((-1, 1, WINDOW_SIZE, 1))  # (6888, 1, 8, 1)
    print(np.shape(x[0]))  # (6, 8, 3)
    print(np.shape(x[:,0])) # (6888, 8, 3)
    print(np.shape(x[0][0]))  # (8, 3)
    # print(np.shape(x[0][0][0]))  # (3,)

    print(x[0][0])
    print(x[1][0])

    # print(np.shape(x))  # (6888, 6, 8, 3)
    # print(np.shape(y))  # (6888, 1)
    # print(np.shape(l))  # (6888, 1, 8, 1)
    return x, y, l

data, label, feature = split_data_load(data_type='raw', index='train')


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


# x, y, l = featured_data_load('raw', 'train')

