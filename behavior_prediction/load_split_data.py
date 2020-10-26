import numpy as np

PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
RAW_LABELS = {
  'Hold': 0,
  'Rest': 1,
  'Preparation': 2,
  'Retraction': 3,
  'Stroke': 4
}


def data_load(data_type, index):
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
            x.append(temp)
        for value in label:
            y.append(int(value.rstrip()))

    print(np.shape(x))
    print(np.shape(y))
    print(x[0])
    print(y[0])
    print(np.shape(np.array(x).reshape((-1, 8, 6, 3))))
    print(x[0])
    x = np.array(x).reshape((-1, 8, 6, 3))
    y = np.array(y).reshape((-1, 1))
    return x, y


data, label = data_load(data_type='raw', index='train')
print(data[0])
print(np.shape(data[0]))