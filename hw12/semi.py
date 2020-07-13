import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import FeatureExtractor, LabelPredictor, DomainClassifier

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

target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
feature_extractor.load_state_dict(torch.load('strong1_extractor_model_1000.bin'))
label_predictor = LabelPredictor().cuda()
label_predictor.load_state_dict(torch.load('strong1_predictor_model_1000.bin'))
domain_classifier = DomainClassifier().cuda()
#domain_classifier.load_state_dict(torch.load('extractor_model_300.bin'))

feature_extractor.eval()
label_predictor.eval()
label_dict = {}
for i in range(10):
    label_dict[i] = []

for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    _y = torch.max(class_logits, dim=1)
    y = _y[0].cpu().detach().numpy()
    data = test_data.cpu().detach().numpy()
    for j in range(len(x)):
        label = x[j]
        if len(label_dict[label]) < 500:
            label_dict[label].append((y[j], data[j]))
        elif y[j] > label_dict[label][0][0]:
            del label_dict[label][0]
            label_dict[label].append((y[j], data[j]))
            sorted(label_dict[label], key=lambda k: k[0])

import matplotlib.pyplot as plt

for i in range(len(label_dict[0][0][1])):
    for j in range(len(label_dict[0][0][1][0])):
        print(label_dict[0][0][1][i][j], end=',')
    print()
for i in range(10):
    print('len = ', len(label_dict[i]))
    for j in range(len(label_dict[i])):
        image = label_dict[i][j][1].transpose(1, 2, 0)
        image = (image * 255).astype('uint8')
        cv2.imwrite(f'./real_or_drawing/semi/{i}/{j}.bmp', image)
        

            