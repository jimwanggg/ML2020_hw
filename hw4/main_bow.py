# main.py
import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
# my class
import utils
import train
import test
from preprocess import Preprocess
from data import TwitterDataset
from model import D_Net
from bow import BOW

path_prefix = './ml2020spring-hw4/'

# 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 處理好各個data的路徑
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
testing_data = os.path.join(path_prefix, 'testing_data.txt')

w2v_path = os.path.join(path_prefix, 'w2v_all.model') # 處理word to vec model的路徑

# 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
sen_len = 40
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 30
lr = 0.0001
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
model_dir = path_prefix # model directory for checkpoint model

print("loading data ...") # 把'training_label.txt'跟'training_nolabel.txt'讀進來
train_x, y = utils.load_training_data(train_with_label)
train_x_no_label = utils.load_training_data(train_no_label)
test_x = utils.load_testing_data(testing_data)

# 對input跟labels做預處理
max_len = 1200
b = BOW(max_len=max_len)
b.bow(train_x, test_x)
train_x = b['train']
#import pdb
#pdb.set_trace()
y = [int(label) for label in y]
y = torch.LongTensor(y)

# 製作一個model的對象
model = D_Net(embedding_dim=max_len, num_layers=1)
model = model.to(device) # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

# 把data分為training data跟validation data(將一部份training data拿去當作validation data)
X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

# 把data做成dataset供dataloader取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 0)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 0)

# 開始訓練
train.training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)