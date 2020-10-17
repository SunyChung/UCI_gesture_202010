import numpy as np

PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_data(data_name, data_type, return_type):
    train_test_path = './dataset/train_test/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '.csv', 'r')
        label = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '_label.txt', 'r')

        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for value in line:
                temp = temp + [float(value)]
            # print(np.shape(temp))
            x.append(temp)

        for l in label:
            y.append(data_dict[l.rstrip()])
    # for Conv1D input :
    # x = np.array(x).reshape((-1, WINDOW_SIZE, 18))
    # for Conv2D input :
    # x = np.array(x).reshape((-1, WINDOW_SIZE, 18, 1))
    if return_type == '1D':
        if data_name == 'raw':
            x = np.array(x).reshape((-1, WINDOW_SIZE, 18))
        else:
            x = np.array(x).reshape((-1, WINDOW_SIZE, 32))
    else:
        if data_name == 'raw':
            x = np.array(x).reshape((-1, WINDOW_SIZE, 18, 1))
        else:
            x = np.array(x).reshape((-1, WINDOW_SIZE, 32, 1))
    y = np.array(y).reshape((-1, 1))
    print(np.shape(x))
    print(np.shape(y))
    return x, y


def load_data_with_label(data_name, data_type):
    train_test_path = './dataset/train_test_with_label/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '.csv', 'r')
        label = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '_label.txt',
                     'r')

        for line in data:
            line = line.rstrip().split(',')
            temp = []
            for value in line:
                if value in line[:-1]:
                    temp = temp + [float(value)]
                else:
                    temp = temp + [data_dict[value.rstrip()]]
            x.append(temp)

        for l in label:
            y.append(data_dict[l.rstrip()])

    # print(np.shape(x))
    # print(np.shape(y))
    x = np.array(x).reshape((-1, WINDOW_SIZE, 19))
    y = np.array(y).reshape((-1, WINDOW_SIZE, 1))
    # print(np.shape(x))
    # print(np.shape(y))
    return x, y

# data, label = load_data_with_label('raw', 'train')
# print(data)
# print(label)