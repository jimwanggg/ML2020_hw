from torch.utils.data import Dataset
import os
import cv2

class ImgDataset(Dataset):
    def __init__(self, src_transform, target_transform, root, train = True):
        self.src_transform = src_transform
        self.target_transform = target_transform
        self.train = train
        self.x = []
        self.y = []
        if train == True:
            for i in range(10):
                folder = os.path.join(root, f'train_data/{i}')
                for j, file in enumerate(os.listdir(folder)):
                    img = cv2.imread(os.path.join(folder, file))
                    #img = img.transpose(1, 2, 0)
                    self.y.append(i)
                    self.x.append(img)
            for i in range(10):
                folder = os.path.join(root, f'semi/{i}')
                for j, file in enumerate(os.listdir(folder)):
                    img = cv2.imread(os.path.join(folder, file))
                    #img = img.transpose(1, 2, 0)
                    self.y.append(i)
                    self.x.append(img)
        else:
            folder = os.path.join(root, f'test_data/0')
            for j, file in enumerate(os.listdir(folder)):
                img = cv2.imread(os.path.join(folder, file))
                #img = img.transpose(1, 2, 0)
                self.x.append(img)

    def __getitem__(self, index):
        if self.train:
            if index < 5000:
                out = self.src_transform(self.x[index])
            else:
                out = self.target_transform(self.x[index])
            return out, self.y[index]
        else:
            return self.target_transform(self.x[index]), -1

    def __len__(self):
        return len(self.x)
