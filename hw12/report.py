from sklearn.decomposition import PCA
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import math
import matplotlib.pyplot as plt

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor, LabelPredictor, DomainClassifier, DomainClassifier2


source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # 重新將np.array 轉回 skimage.Image
    transforms.ToPILImage(),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),#, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),#, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
feature_extractor.load_state_dict(torch.load('noDANN_extractor_1000.bin'))

feature_extractor.eval()

result_test = []
result_train = []

for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    feature = feature_extractor(test_data).cpu().detach().numpy()

    #x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result_test.append(feature)

result_test = np.concatenate(result_test)
print('done 1')

for i, (test_data, _) in enumerate(source_dataloader):
    test_data = test_data.cuda()

    feature = feature_extractor(test_data).cpu().detach().numpy()

    #x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result_train.append(feature)

result_train = np.concatenate(result_train)
print('done 2')
k = np.concatenate((result_train, result_test))

pca_total = PCA(n_components=2, random_state=0).fit(k)
A = pca_total.transform(result_train)
B = pca_total.transform(result_test)
plt.scatter(B[:,0], B[:,1], c='r', s = 0.3,label = 'target')
plt.scatter(A[:,0], A[:,1], s = 0.3, label = 'source')
plt.show()

