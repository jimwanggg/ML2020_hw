import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ImgDataset(Dataset):
    def __init__(self, x, y=None, mode = 'train'):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        #training 時做 data augmentation
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
            transforms.RandomRotation(15), #隨機旋轉圖片
            transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
        ])
        #testing 時不需做 data augmentation
        test_transform = transforms.Compose([
            transforms.ToPILImage(),                                    
            transforms.ToTensor(),
        ])
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = train_transform if mode == 'train' else test_transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
    def getbatch2(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return images, torch.tensor(labels)