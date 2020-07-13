import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

w = np.load('weight_mine.npy')
mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')

## testing ##
testdata = pd.read_csv(input_file, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
data = int(test_data.shape[0]/18)
## process <0 data ##
for i in range(18*data):
    _sum = 0
    _times = 0
    for j in range(9):
        if float(test_data[i][j]) >= 0:
            _sum += float(test_data[i][j])
            _times += 1
    _sum /= _times
    str_sum = str(_sum)
    for j in range(9):
        if float(test_data[i][j]) < 0:
            test_data[i][j] = str_sum

after_list = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]


## MINE ##
test_x_after = np.empty([data, len(after_list)*9], dtype = float)
for i in range(data):
    for j in range(len(after_list)):
        test_x_after[i, 9 * j: 9 * (j + 1)] = test_data[18 * i + after_list[j], :].reshape(1, -1)
for i in range(len(test_x_after)):
    for j in range(len(after_list)):
        for k in range(9):
            index = after_list[j] * 9 + k
            if std_x[index] != 0:
                test_x_after[i][j*9 + k] = (test_x_after[i][j*9 + k] - mean_x[index]) / std_x[index]
test_x_after = np.concatenate((np.ones([data, 1]), test_x_after), axis = 1).astype(float)

ans_y_mine = np.dot(test_x_after, w)

with open(output_file, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(data):
        row = ['id_' + str(i), ans_y_mine[i][0]]
        csv_writer.writerow(row)

