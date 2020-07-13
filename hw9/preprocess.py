import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list2 = np.transpose(image_list, (0, 3, 1, 2))
    image_list2 = (image_list2 / 255.0) * 2 - 1
    image_list2 = image_list2.astype(np.float32)
    return image_list2, image_list



class Image_Dataset(Dataset):
    def __init__(self, image_list, transform = None):
        self.image_list = image_list
        self.test_transform = self.trans('test')
        self.train_transform = self.trans('train')
    def __len__(self):
        return len(self.image_list)

    def __change(self, img):
        return torch.add(torch.mul(img, 2), -1)

    def __noise(self, img):
        img = -2 * torch.rand(img.shape) + 1
        img = torch.clamp(img, 0, 1)
        return img

    def trans(self, mode):
        if mode == 'test':
            ret = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                #transforms.Lambda(lambda img:self.__change(img))
            ])
        else:
            ret = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                #transforms.RandomApply([lambda img:self.__noise(img)], p = 0.3),
                #transforms.Lambda(lambda img:self.__change(img))
            ])
        return ret 
    
    def __getitem__(self, idx):
        images = self.image_list[idx]
        images2 = self.train_transform(images)       
        images3 = self.test_transform(images) 
        return images3, images2


class Image_Dataset2(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images