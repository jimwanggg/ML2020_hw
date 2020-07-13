# main.py
import os
import torch
import argparse
import sys
import numpy as np
import pandas as pd
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
# my class
import utils
import train
import test
from preprocess import Preprocess
from data import TwitterDataset
from model import LSTM_Net, LSTM_Net_BI

def make_avg_list(_list):
    _len = len(_list)
    _total_len = len(_list[0])
    _avg = [0] * _total_len
    for lists in _list:
        for index in range(_total_len):
            _avg[index] += lists[index]
    _avg = [(num/_len) for num in _avg]
    return _avg

if __name__ == '__main__':
    testing_data = sys.argv[1]
    predict = sys.argv[2]
    w2v_path = 'w2v_with_nolabel.model'
    sen_len = 40
    fix_embedding = True # fix embedding during training
    batch_size = 128
    # 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading testing data ...")
    test_x = utils.load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)
    print('load model ...')
    ensemble = []
    model = torch.load('first_strong.model')
    ensemble.append(test.test_prob(batch_size, test_loader, model, device))
    model = torch.load('second_strong.model')
    ensemble.append(test.test_prob(batch_size, test_loader, model, device))
    model = torch.load('third_strong.model')
    ensemble.append(test.test_prob(batch_size, test_loader, model, device))
    model = torch.load('fourth_strong.model')
    ensemble.append(test.test_prob(batch_size, test_loader, model, device))
    model = torch.load('bi1.model')
    ensemble.append(test.test_prob(batch_size, test_loader, model, device))
    avg_list = make_avg_list(ensemble)
    for i in range(len(avg_list)):
        if avg_list[i] <= 0.5:
            avg_list[i] = 0
        else:
            avg_list[i] = 1

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":avg_list})
    print("save csv ...")
    tmp.to_csv(predict, index=False)
    print("Finish Predicting")