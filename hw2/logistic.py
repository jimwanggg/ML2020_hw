import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
X_feature_fpath = './feature_250.txt'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    #next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f])
    print(X_train)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
    print(Y_train)
with open(X_test_fpath) as f:
    #next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f])
    print(X_test)

with open(X_feature_fpath) as f:
    #next(f)
    X_feature = np.array([line.split(',') for line in f], dtype=int)
    print(X_feature)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _train_dev_split_cross(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    folds = int(1/dev_ratio)
    print(folds)
    fold_size = int(len(X) * (dev_ratio))
    X_out = []
    Y_out = []
    X_cross = []
    Y_cross = []
    for i in range(folds):
        X_out.append(np.delete(X, slice(i*fold_size, (i+1)*fold_size), 0))
        Y_out.append(np.delete(Y, slice(i*fold_size, (i+1)*fold_size), 0))
        X_cross.append(np.array(X)[i*fold_size:(i+1)*fold_size])
        Y_cross.append(np.array(Y)[i*fold_size:(i+1)*fold_size])
    return X_out, Y_out, X_cross, Y_cross

print(X_train.shape)


# extract feature
X_train = np.delete(X_train, X_feature, 1)
X_test = np.delete(X_test, X_feature, 1)
print(X_train)


# feature engineering
print(X_train.shape)
#X_train = X_train[:, np.logical_and(X_train[0] != ' Not in universe', X_train[0] != ' ?')]
X_train = X_train[:, np.logical_and(X_train[0] != ' 94', X_train[0] != ' 95')]
#X_train = X_train[:, col_mask]
week = 0
capital_loss = 0
capital_gain = 0
wage_hour = 0
div = 0
emp = 0

for i in range(X_train.shape[1]):
    if X_train[0][i] == 'weeks worked in year':
        print('week = ', i)
        week = i
    elif X_train[0][i] == 'capital losses':
        print('losses = ', i)
        capital_loss = i
    elif X_train[0][i] == 'capital gains':
        print('gain = ', i)
        capital_gain = i
    elif X_train[0][i] == 'wage per hour':
        print('wage = ', i)
        wage_hour = i
    elif X_train[0][i] == 'dividends from stocks':
        print('dividends = ', i)
        div = i 
    elif X_train[0][i] == 'num persons worked for employer':
        print('emp = ', i)
        emp = i 

print(X_train.shape)
X_train = X_train[1:].astype(float)
#X_test = X_test[:, np.logical_and(X_test[0] != ' Not in universe', X_test[0] != ' ?')]
X_test = X_test[:, np.logical_and(X_test[0] != ' 94', X_test[0] != ' 95')]
X_test = X_test[1:].astype(float)
print(X_test)

'''
 # education level list
edu_level = [2, 8, 1, 10, 13, 4, 7, 14, 15, 10, 3, 16, 10, 9, 0, 6, 5]
# 1234, 12, <1, Some college but no degree, MS, 78, 11, BA, Prof school degree, Associates degree-academic program, 56, phd, Associates degree-occup /vocational, High school graduate, 0, 10, 9

'''
age_list = [15, 24, 29, 34, 39, 44, 54, 59, 64, 69, 100]
age_encode = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2]

wage_list = [0, 1500, 2000, 2500, 100000]
wage_encode = [1, 0, 2, 3, 4]

capital_gain_list = [0, 3000, 4000, 7000, 10000, 1000000]
capital_gain_encode = [2, 0, 1, 3, 4, 5]

capital_loss_list = [1800, 2400, 1000000]
capital_loss_encode = [0, 1, 2]

div_list = [0, 900, 4500, 9900, 22500, 1000000]
div_encode = [0, 1, 2, 3, 4, 5]

emp_list = [0, 3, 4, 5, 1000000]
emp_encode = [0, 1, 2, 3, 4]

week_list = [35, 45, 50, 1000000]
week_encode = [0, 1, 2, 3]

 # age
arr_age = np.zeros([X_train.shape[0], 7])
arr_age_test = np.zeros([X_test.shape[0], 7])
# wage
arr_wage = np.zeros([X_train.shape[0], 5])
arr_wage_test = np.zeros([X_test.shape[0], 5])
# capital gain
arr_gain = np.zeros([X_train.shape[0], 6])
arr_gain_test = np.zeros([X_test.shape[0], 6])
# capital loss
arr_loss = np.zeros([X_train.shape[0], 3])
arr_loss_test = np.zeros([X_test.shape[0], 3])
# divendends
arr_div = np.zeros([X_train.shape[0], 6])
arr_div_test = np.zeros([X_test.shape[0], 6])
# num of employers
arr_emp = np.zeros([X_train.shape[0], 5])
arr_emp_test = np.zeros([X_test.shape[0], 5])
# week
arr_week = np.zeros([X_train.shape[0], 4])
arr_week_test = np.zeros([X_test.shape[0], 4])

for i in range(X_train.shape[0]):
    # age
    age = X_train[i][0]
    for j in range(len(age_list)):
        if age <= age_list[j]:
            for k in range(age_encode[j]):
                arr_age[i][k] = 1
            break
    # wage
    wage = X_train[i][wage_hour]
    for j in range(len(wage_list)):
        if wage <= wage_list[j]:
            for k in range(wage_encode[j]):
                arr_wage[i][k] = 1
            break
    # gain
    gain = X_train[i][capital_gain]
    for j in range(len(capital_gain_list)):
        if gain <= capital_gain_list[j]:
            #arr_gain[i][capital_gain_encode[j]] = 1
            for k in range(capital_gain_encode[j]):
                arr_gain[i][k] = 1
            break
    # loss
    loss = X_train[i][capital_loss]
    for j in range(len(capital_loss_list)):
        if loss <= capital_loss_list[j]:
            #arr_loss[i][capital_loss_encode[j]] = 1
            for k in range(capital_loss_encode[j]):
                arr_loss[i][k] = 1
            break
    # div
    diven = X_train[i][div]
    for j in range(len(div_list)):
        if diven <= div_list[j]:
            #arr_div[i][div_encode[j]] = 1
            for k in range(div_encode[j]):
                arr_div[i][k] = 1
            break
    # num_emp
    num_emp = X_train[i][emp]
    for j in range(len(emp_list)):
        if num_emp <= emp_list[j]:
            #arr_emp[i][emp_encode[j]] = 1
            for k in range(emp_encode[j]):
                arr_emp[i][k] = 1
            break
    # week
    num_week = X_train[i][week]
    for j in range(len(week_list)):
        if num_week <= week_list[j]:
            #arr_week[i][week_encode[j]] = 1
            for k in range(week_encode[j]):
                arr_week[i][k] = 1
            break
            
    
for i in range(X_test.shape[0]):
    age = X_test[i][0]
    for j in range(len(age_list)):
        if age <= age_list[j]:
            for k in range(age_encode[j]):
                arr_age_test[i][k] = 1
            break  
    # wage
    wage = X_test[i][wage_hour]
    for j in range(len(wage_list)):
        if wage <= wage_list[j]:
            for k in range(wage_encode[j]):
                arr_wage_test[i][k] = 1
            break
    # gain
    gain = X_test[i][capital_gain]
    for j in range(len(capital_gain_list)):
        if gain <= capital_gain_list[j]:
            #arr_gain_test[i][capital_gain_encode[j]] = 1
            for k in range(capital_gain_encode[j]):
                arr_gain_test[i][k] = 1
            break
    # loss
    loss = X_test[i][capital_loss]
    for j in range(len(capital_loss_list)):
        if loss <= capital_loss_list[j]:
            #arr_loss[i][capital_loss_encode[j]] = 1
            for k in range(capital_loss_encode[j]):
                arr_loss_test[i][k] = 1
            break
    # div
    diven = X_test[i][div]
    for j in range(len(div_list)):
        if diven <= div_list[j]:
            #arr_div[i][div_encode[j]] = 1
            for k in range(div_encode[j]):
                arr_div_test[i][k] = 1
            break
    # num_emp
    num_emp = X_test[i][emp]
    for j in range(len(emp_list)):
        if num_emp <= emp_list[j]:
            #arr_emp[i][emp_encode[j]] = 1
            for k in range(emp_encode[j]):
                arr_emp_test[i][k] = 1
            break
    # week
    num_week = X_test[i][week]
    for j in range(len(week_list)):
        if num_week <= week_list[j]:
            #arr_week[i][week_encode[j]] = 1
            for k in range(week_encode[j]):
                arr_week_test[i][k] = 1
            break

# age
X_train = np.concatenate((arr_age, np.delete(X_train, 0, 1)), axis = 1)
X_test = np.concatenate((arr_age_test, np.delete(X_test, 0, 1)), axis = 1)
print(X_train.shape)

# wage
X_train_wage = np.concatenate((X_train[:, 0:wage_hour+6], arr_wage), axis = 1)
X_train = np.concatenate((X_train_wage, X_train[:, (wage_hour+7):]), axis = 1)
X_test_wage = np.concatenate((X_test[:, 0:wage_hour+6], arr_wage_test), axis = 1)
X_test = np.concatenate((X_test_wage, X_test[:, (wage_hour+7):]), axis = 1)
print(X_train.shape)

# gain
X_train_gain = np.concatenate((X_train[:, 0:capital_gain+6+4], arr_gain), axis = 1)
X_train = np.concatenate((X_train_gain, X_train[:, (capital_gain+7+4):]), axis = 1)
X_test_gain = np.concatenate((X_test[:, 0:capital_gain+6+4], arr_gain_test), axis = 1)
X_test = np.concatenate((X_test_gain, X_test[:, (capital_gain+7+4):]), axis = 1)
print(X_train.shape)

# loss
X_train_loss = np.concatenate((X_train[:, 0:capital_loss+6+4+5], arr_loss), axis = 1)
X_train = np.concatenate((X_train_loss, X_train[:, (capital_loss+7+4+5):]), axis = 1)
X_test_loss = np.concatenate((X_test[:, 0:capital_loss+6+4+5], arr_loss_test), axis = 1)
X_test = np.concatenate((X_test_loss, X_test[:, (capital_loss+7+4+5):]), axis = 1)
print(X_train.shape)

# div
X_train_div = np.concatenate((X_train[:, 0:div+6+4+5+2], arr_div), axis = 1)
X_train = np.concatenate((X_train_div, X_train[:, (div+7+4+5+2):]), axis = 1)
X_test_div = np.concatenate((X_test[:, 0:div+6+4+5+2], arr_div_test), axis = 1)
X_test = np.concatenate((X_test_div, X_test[:, (div+7+4+5+2):]), axis = 1)
print(X_train.shape)

# emp
X_train_emp = np.concatenate((X_train[:, 0:emp+6+4+5+2+5], arr_emp), axis = 1)
X_train = np.concatenate((X_train_emp, X_train[:, (emp+7+4+5+2+5):]), axis = 1)
X_test_emp = np.concatenate((X_test[:, 0:emp+6+4+5+2+5], arr_emp_test), axis = 1)
X_test = np.concatenate((X_test_emp, X_test[:, (emp+7+4+5+2+5):]), axis = 1)
print(X_train.shape)

# week
X_train_week = np.concatenate((X_train[:, 0:week+6+4+5+2+5+4], arr_week), axis = 1)
X_train = np.concatenate((X_train_week, X_train[:, (week+7+4+5+2+5+4):]), axis = 1)
X_test_week = np.concatenate((X_test[:, 0:week+6+4+5+2+5+4], arr_week_test), axis = 1)
X_test = np.concatenate((X_test_week, X_test[:, (week+7+4+5+2+5+4):]), axis = 1)
print(X_train.shape)

norm_col = []
for i in range(X_train.shape[1]):
    for j in range(X_train.shape[0]):
        if X_train[j][i] < 0 or X_train[j][i] > 1:
            norm_col.append(i)
            print(i)
            print(X_train[j][i])
            break


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True, specified_column = norm_col)
X_test, _, _= _normalize(X_test, train = False, specified_column = norm_col, X_mean = X_mean, X_std = X_std)

np.save('X_mean_best', X_mean)
np.save('X_std_best', X_std)

# Split data into training set and development (validation) set
dev_ratio = 0.1
folds_times = int(1/dev_ratio)

#X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
X_cross, Y_cross, X_dev_cross, Y_dev_cross = _train_dev_split_cross(X_train, Y_train, dev_ratio = dev_ratio)


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label, w, lamb = 0):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred)) + lamb * (np.sum(w))**2
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

data_dim = X_cross[0].shape[1]

train_least_loss = []
test_size = X_test.shape[0]
w_last = np.zeros((data_dim,))
b_last = np.zeros((1,))
w_avg = np.zeros((data_dim,))
b_avg = np.zeros((1,))
min_loss = 100000000
total_loss = 0
total_acc = 0
## my cross validation ##
for _fold in range(folds_times):
    train_size = X_cross[_fold].shape[0]
    dev_size = X_dev_cross[_fold].shape[0]

    w = np.zeros((data_dim,))
    b = np.zeros((1,))

    max_iter = 500
    batch_size = 128
    learning_rate = 0.05
    print(_fold)
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []
    step = 1

    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train_cur, Y_train_cur = _shuffle(X_cross[_fold], Y_cross[_fold])
        if epoch % 100 == 0:
            print(max_iter)
        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train_cur[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train_cur[idx*batch_size:(idx+1)*batch_size]

            # Compute the gradient
            w_grad, b_grad = _gradient(X, Y, w, b)

            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate/np.sqrt(step) * w_grad
            b = b - learning_rate/np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_cross[_fold], w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_cross[_fold]))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_cross[_fold], w) / train_size)

        y_dev_pred = _f(X_dev_cross[_fold], w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev_cross[_fold]))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev_cross[_fold], w) / dev_size)

    total_loss += dev_loss[-1]
    total_acc += dev_acc[-1]
    if dev_loss[-1] < min_loss:
        w_last = w
        b_last = b
        min_loss = dev_loss[-1]
        print(_fold, ' fold is better !')
    w_avg = w_avg + w
    b_avg = b_avg + b
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))
    
w_avg = w_avg / folds_times
b_avg = b_avg / folds_times
avg_loss = total_loss / folds_times
avg_acc = total_acc / folds_times
print('avg_loss = ', avg_loss)
print('avg_acc = ', avg_acc)
print('avg w = ', w_avg)
print('avg b = ', b_avg)
np.save('w_best', w_avg)
np.save('b_best', b_avg)

# Predict testing labels
predictions = _predict(X_test, w_avg, b_avg)
with open(output_fpath.format('logistic_30_8_01_encode_noweek'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
