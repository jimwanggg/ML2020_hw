import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

input_file = './ml2020spring-hw1/train.csv'
output_file = './ml2020spring-hw1/test.csv'

data = pd.read_csv(input_file, encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

#print(raw_data)
month_data = {}

## process <0 data ##
for i in range(18*20*12):
    _sum = 0
    _times = 0
    for j in range(24):
        if float(raw_data[i][j]) >= 0:
            _sum += float(raw_data[i][j])
            _times += 1
    _sum /= _times
    str_sum = str(_sum)
    for j in range(24):
        if float(raw_data[i][j]) < 0:
            raw_data[i][j] = str_sum

for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample


x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
#print(x)
#print(y)


mean_x = np.mean(x, axis = 0) #18 * 9
std_x = np.std(x, axis = 0) #18 * 9
np.save('mean_x.npy', mean_x)
np.save('std_x.npy', std_x)
'''
for _ in range(17, -1, -1):
    plt.plot(x_f[_], '.')
    plt.plot(y, 'r.')
    plt.show()
'''

for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    #for j in range(18):

## split features ##
x_f = np.split(x, 18, axis = 1)
print(x_f)


x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

'''
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))
'''

## ADAgrad ##
dim = 18 * 9 + 1
dim_feature = 9 + 1
dim_after = 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
x_f_after = np.ones([12 * 471, 1])
learning_rate = 10
learning_rate_f = 3
iter_time = 30000
iter_time_f = 3000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
after_list = []
print(x)

## testing different features ##
for _ in range(18):
    x_f[_] = np.concatenate((np.ones([12 * 471, 1]), x_f[_]), axis = 1).astype(float)
print(x_f[0][:, 1:].shape)

for f in range(18):
    w_feature = np.zeros([dim_feature, 1])
    adagrad_f = np.zeros([dim_feature, 1])
    loss = 0
    for t in range(iter_time_f):
        loss = np.sqrt(np.sum(np.power(np.dot(x_f[f], w_feature) - y, 2))/471/12)#rmse
        gradient = 2 * np.dot(x_f[f].transpose(), np.dot(x_f[f], w_feature) - y) #dim*1
        adagrad_f += gradient ** 2
        w_feature = w_feature - learning_rate_f * gradient / np.sqrt(adagrad_f + eps)
    print('loss of feature', f, '=', loss)
    if loss < 16:
        x_f_after = np.concatenate((x_f_after, x_f[f][:, 1:]), axis = 1).astype(float)
        dim_after += 9
        print('add feature ', f)
        after_list.append(f)
        print(dim_after)

## TA's work ##
loss_TA = 0
for t in range(iter_time):
    loss_TA = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%1000==0):
        print('TAs = ' + str(t) + ":" + str(loss_TA))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

## my work ##
w_after = np.zeros([dim_after, 1])
adagrad_after = np.zeros([dim_after, 1])
loss_mine = 0
for t in range(iter_time):
    loss_mine = np.sqrt(np.sum(np.power(np.dot(x_f_after, w_after) - y, 2))/471/12)#rmse
    if(t%1000==0):
        print('mine = ' + str(t) + ":" + str(loss_mine))
    gradient = 2 * np.dot(x_f_after.transpose(), np.dot(x_f_after, w_after) - y) #dim*1
    adagrad_after += gradient ** 2
    w_after = w_after - learning_rate * gradient / np.sqrt(adagrad_after + eps)

print('TA loss = ' + str(loss_TA))
print('my loss = ' + str(loss_mine))
'''
np.save('weight.npy', w)
np.save('weight_mine.npy', w_after)
'''
## testing ##
testdata = pd.read_csv(output_file, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

## process <0 data ##
for i in range(18*240):
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
test_x = np.empty([240, 18*9], dtype = float)
print(len(test_x))
print(len(test_x[0]))
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

## MINE ##
test_x_after = np.empty([240, len(after_list)*9], dtype = float)
for i in range(240):
    for j in range(len(after_list)):
        test_x_after[i, 9 * j: 9 * (j + 1)] = test_data[18 * i + after_list[j], :].reshape(1, -1)
for i in range(len(test_x_after)):
    for j in range(len(after_list)):
        for k in range(9):
            index = after_list[j] * 9 + k
            if std_x[index] != 0:
                test_x_after[i][j*9 + k] = (test_x_after[i][j*9 + k] - mean_x[index]) / std_x[index]
test_x_after = np.concatenate((np.ones([240, 1]), test_x_after), axis = 1).astype(float)
'''
## predict ##
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

w2 = np.load('weight_mine.npy')
ans_y_mine = np.dot(test_x_after, w2)

## save file ##
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    #print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        #print(row)

with open('submit_mine.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    #print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y_mine[i][0]]
        csv_writer.writerow(row)
        #print(row)
'''