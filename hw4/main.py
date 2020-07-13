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

if __name__ == '__main__':
    path_prefix = './'
    w2v_model = 'w2v_with_nolabel.model'
    #testing_data = os.path.join(path_prefix, 'testing_data.txt')
    training_pos = sys.argv[1]
    training_nolabel_pos = sys.argv[2]
    # 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個 data 的路徑
    train_with_label = os.path.join(path_prefix, training_pos)
    train_no_label = os.path.join(path_prefix, training_nolabel_pos)

    w2v_path = os.path.join(path_prefix, w2v_model) # 處理 word to vec model 的路徑

    # 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
    sen_len = 40
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 50
    lr = 0.0001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = path_prefix # model directory for checkpoint model

    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = utils.load_training_data(train_with_label)
    train_x_no_label = utils.load_training_data(train_no_label)

    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_2 = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)
    print('embedding size = ', embedding.size())
    # 製作一個 model 的對象
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=200, num_layers=2, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x_2[:180000], train_x_2[180000:], y[:180000], y[180000:]
    #X_train, X_val, y_train, y_val = train_x_2[:20000], train_x_2[20000:40000], y[:20000], y[20000:40000]
    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把 data 轉成 batch of tensors
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

    # test no label data
    print('start predict in no label data...')
    preprocess = Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_nolabel_x = preprocess.sentence_word2idx()
    train_nolabel_dataset = TwitterDataset(X=train_nolabel_x, y=None)
    train_nolabel_loader = torch.utils.data.DataLoader(dataset = train_nolabel_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)
    print('load model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    y_nolabel, y_good_indices = test.testing(batch_size, train_nolabel_loader, model, device, threshold=0.8)
    total_train_num = 200000 + len(y_good_indices)
    val_num = int(0.1 * total_train_num)
    print('val_num = {num}, train_num = {val}'.format(num = val_num, val = total_train_num-val_num))
    print('good indices = ', len(y_good_indices))
    y_nolabel = preprocess.labels_to_tensor(y_nolabel)
    y_good_indices2 = preprocess.labels_to_tensor(y_good_indices)
    train_nolabel_x = train_nolabel_x.index_select(0, y_good_indices2)
    y_nolabel = y_nolabel.index_select(0, y_good_indices2)

    # train again with all data
    print('start processing ALL...')
    print('train_nolabel = ', train_nolabel_x.size())
    print('y_nolabel = ', y_nolabel.size())
    # preprocessing again...
    print('preprocessing...')
    new_no_label_x = [train_x_no_label[index] for index in y_good_indices]
    print('len of preprocessing', len(train_x + new_no_label_x))
    preprocess = Preprocess(train_x+new_no_label_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    all_x = preprocess.sentence_word2idx()

    #all_x = torch.cat((train_x, train_nolabel_x), 0)

    all_y = torch.cat((y, y_nolabel))
    #all_y = torch.cat((y[:20000],y_nolabel))
    print('all_x = ', all_x.size())
    print('all_y = ', all_y.size())  
    print('embedding size = ', embedding.size())  

    # 製作一個 model 的對象
    model2 = LSTM_Net(embedding, embedding_dim=250, hidden_dim=200, num_layers=2, dropout=0.5, fix_embedding=fix_embedding)
    model2 = model2.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）  
    X_all, X_all_val, Y_all, Y_all_val = all_x[val_num:], all_x[:val_num], all_y[val_num:], all_y[:val_num]
    all_dataset = TwitterDataset(X=all_x, y=all_y)
    val_all_dataset = TwitterDataset(X=X_all_val, y=Y_all_val)
    all_loader = torch.utils.data.DataLoader(dataset = all_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 0)
    all_val_loader = torch.utils.data.DataLoader(dataset = val_all_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)
    print('start training ALL...')  
    epoch = 30


    train.training(batch_size, epoch, lr, model_dir, all_loader, all_val_loader, model2, device)

'''
    # 開始測試模型並做預測
    
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
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs, bad_output = test.testing(batch_size, test_loader, model, device)

    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")

    # 以下是使用 command line 上傳到 Kaggle 的方式
    # 需要先 pip install kaggle、Create API Token，詳細請看 https://github.com/Kaggle/kaggle-api 以及 https://www.kaggle.com/code1110/how-to-submit-from-google-colab
    # kaggle competitions submit [competition-name] -f [csv file path]] -m [message]
    # e.g., kaggle competitions submit ml-2020spring-hw4 -f output/predict.csv -m "......"
'''