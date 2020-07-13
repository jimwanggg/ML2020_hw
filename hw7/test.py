import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import time
from torch.nn import functional as F
import sys
import pickle
from data import MyDataset
from model import StudentNet, MobileNetv2, TANet

trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

data_dir = sys.argv[1]
out_file = sys.argv[2]

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        os.path.join(data_dir, mode),
        transform=trainTransform if mode == 'training' else testTransform,
        label = False if mode == 'testing' else True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training')
    )

    return dataloader

batch_size = 128
test_loader = get_dataloader('testing', batch_size=batch_size)

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            #print(min_val, max_val)
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

model = 'model.pkl'


k = decode8(model)
net = StudentNet().cuda()
net.load_state_dict(k)
net.eval()

prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred, X = data
        test_pred = net(test_pred.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis = 1)
        for y in test_label:
            prediction.append(y)

with open(out_file, 'w') as f:
    f.write('Id,Label\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))



