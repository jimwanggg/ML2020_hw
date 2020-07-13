import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

w = np.load('weight.npy')
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

## TA ##
test_x = np.empty([data, 18*9], dtype = float)
print(len(test_x))
print(len(test_x[0]))
for i in range(data):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([data, 1]), test_x), axis = 1).astype(float)

ans_y = np.dot(test_x, w)

with open(output_file, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    #print(header)
    csv_writer.writerow(header)
    for i in range(data):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        #print(row)