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


def data_load(data_type, index, return_type):
    data_dir = './dataset/per_win_train_test/'
    if data_type == 'raw':
        data_dict = RAW_LABELS
    else:
        data_dict = PRO_LABELS

    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '.csv', 'r')
        label = open(data_dir + str(prefix) + '_' + str(data_type) + '_' + str(index) + '_label.txt', 'r')

        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for value in line:
                temp = temp + [float(value)]
            x.append(temp)

        for value in label:
            y.append(data_dict[value.rstrip()])

    if return_type == '1D':
        if data_type == 'raw':
            x = np.array(x).reshape((-1, WINDOW_SIZE, 18))
        else:
            x = np.array(x).reshape((-1, WINDOW_SIZE, 32))
        y = np.array(y).reshape((-1, 1))

    elif return_type == '2D':
        if data_type == 'raw':
            x = np.array(x).reshape((-1, WINDOW_SIZE, 18, 1))
        else:
            x = np.array(x).reshape((-1, WINDOW_SIZE, 32, 1))
        y = np.array(y).reshape((-1, 1))

    elif return_type == '3D':
        if data_type == 'raw':
            x = np.array(x).reshape((-1, 1, WINDOW_SIZE, 18, 1))
        else:
            x = np.array(x).reshape((-1, 1, WINDOW_SIZE, 32, 1))
        y = np.array(y).reshape((-1, 1, 1, 1))

    elif return_type == 'split':
        x = np.array(x)
        print(np.shape(x))
        print(np.shape(x[1]))
        print(x[0])
        print(y[0])
        if data_type == 'raw':
            x = np.split(x, 6, axis=1)  # (6, 55104, 3)
            x = np.array(x).reshape((6, -1, 8, 3))
        y = np.array(y).reshape((-1, 1))
        print(x[0][0])
        print(x[1][0])
        print(x[2][0])
        print(x[-1][0])
        #print(x[1])

    else:  # LSTM
        if data_type == 'raw':
            x = np.array(x).reshape((-1, 8, 18))
        else:
            x = np.array(x).reshape((-1, 8, 32))
        y = np.array(y).reshape((-1, 1))

    print('data shapes')
    print(np.shape(x))
    print(np.shape(y))
    return x, y


data, label = data_load(data_type='raw', index='train', return_type='split')
# data, label = per_win_data_load(data_type='raw', index='test', return_type='1D')


print(data[0])
print(np.shape(data[0]))
print(data[0][0])
print(np.shape(data[0][0]))