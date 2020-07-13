import numpy as np
import matplotlib.pyplot as plt
import sys

input_path = sys.argv[1]
out_path = sys.argv[2]

w = np.load('w_best.npy')
b = np.load('b_best.npy')

print(w.shape)

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


with open(input_path) as f:
    #next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f])

with open('feature_250.txt') as f:
    #next(f)
    X_feature = np.array([line.split(',') for line in f], dtype=int)
    print(X_feature)

week = 248
capital_loss = 153
capital_gain = 152
wage_hour = 92
div = 154
emp = 200

X_test = np.delete(X_test, X_feature, 1)

X_test = X_test[:, np.logical_and(X_test[0] != ' 94', X_test[0] != ' 95')]
X_test = X_test[1:].astype(float)

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
arr_age_test = np.zeros([X_test.shape[0], 7])
# wage
arr_wage_test = np.zeros([X_test.shape[0], 5])
# capital gain
arr_gain_test = np.zeros([X_test.shape[0], 6])
# capital loss
arr_loss_test = np.zeros([X_test.shape[0], 3])
# divendends
arr_div_test = np.zeros([X_test.shape[0], 6])
# num of employers
arr_emp_test = np.zeros([X_test.shape[0], 5])
# week
arr_week_test = np.zeros([X_test.shape[0], 4])

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
X_test = np.concatenate((arr_age_test, np.delete(X_test, 0, 1)), axis = 1)
print(X_test.shape)

# wage
X_test_wage = np.concatenate((X_test[:, 0:wage_hour+6], arr_wage_test), axis = 1)
X_test = np.concatenate((X_test_wage, X_test[:, (wage_hour+7):]), axis = 1)
print(X_test.shape)

# gain
X_test_gain = np.concatenate((X_test[:, 0:capital_gain+6+4], arr_gain_test), axis = 1)
X_test = np.concatenate((X_test_gain, X_test[:, (capital_gain+7+4):]), axis = 1)
print(X_test.shape)

# loss
X_test_loss = np.concatenate((X_test[:, 0:capital_loss+6+4+5], arr_loss_test), axis = 1)
X_test = np.concatenate((X_test_loss, X_test[:, (capital_loss+7+4+5):]), axis = 1)
print(X_test.shape)

# div
X_test_div = np.concatenate((X_test[:, 0:div+6+4+5+2], arr_div_test), axis = 1)
X_test = np.concatenate((X_test_div, X_test[:, (div+7+4+5+2):]), axis = 1)
print(X_test.shape)

# emp
X_test_emp = np.concatenate((X_test[:, 0:emp+6+4+5+2+5], arr_emp_test), axis = 1)
X_test = np.concatenate((X_test_emp, X_test[:, (emp+7+4+5+2+5):]), axis = 1)
print(X_test.shape)

# week
X_test_week = np.concatenate((X_test[:, 0:week+6+4+5+2+5+4], arr_week_test), axis = 1)
X_test = np.concatenate((X_test_week, X_test[:, (week+7+4+5+2+5+4):]), axis = 1)
print(X_test.shape)

# Predict testing labels
predictions = _predict(X_test, w, b)
with open(out_path, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
