# test.py
# 這個 block 用來對 testing_data.txt 做預測
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device, threshold = 0.5):
    model.eval()
    ret_output = []
    good_predict = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            if i % 1000 == 3:
                print(i)
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze() # 刪除維度為 1 的 dimension
            u = []
            for k in outputs:
                if k >= threshold:
                    u.append(1)
                elif k < 1-threshold:
                    u.append(0)
                else:
                    u.append(2)
            #outputs[outputs >= threshold] = 1 # 大於等於 0.5 為負面
            #outputs[outputs < threshold] = 2 
            #outputs[outputs < (1-threshold)] = 0 # 小於 0.5 為正面
            ret_output += u #outputs.int().tolist()
    for i in range(len(ret_output)):
        if ret_output[i] != 2:
            good_predict.append(i)
    return ret_output, good_predict

def test_prob(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze() # 刪除維度為 1 的 dimension
            ret_output += outputs.tolist()
    return ret_output
