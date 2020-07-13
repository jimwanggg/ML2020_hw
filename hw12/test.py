import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import math
import os
import sys

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor, LabelPredictor, DomainClassifier

import pandas as pd

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)

target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    #transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    #transforms.RandomRotation(15),#, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

directory = sys.argv[1]
prediction = sys.argv[2]

target_dataset = ImageFolder(os.path.join(directory, 'test_data'), transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
feature_extractor.load_state_dict(torch.load('strong1_extractor_model_1000.bin'))
label_predictor = LabelPredictor().cuda()
label_predictor.load_state_dict(torch.load('strong1_predictor_model_1000.bin'))

feature_extractor.eval()
label_predictor.eval()
result = []

for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)
    #print(i)

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(prediction,index=False)