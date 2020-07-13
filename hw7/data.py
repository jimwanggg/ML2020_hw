import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None, label = True):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):            
            if label:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[-2])
            else:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0
            
            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)
        #print(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]

class AllDataset(torch.utils.data.Dataset):

    def __init__(self, folderNames, transform=None, label = True):
        self.transform = transform
        self.data = []
        self.label = []
        for folderName in folderNames:
            for img_path in sorted(glob(folderName + '/*.jpg')):            
                if label:
                    # Get classIdx by parsing image path
                    class_idx = int(re.findall(re.compile(r'\d+'), img_path)[-2])
                else:
                    # if inference mode (there's no answer), class_idx default 0
                    class_idx = 0
                
                image = Image.open(img_path)
                # Get File Descriptor
                image_fp = image.fp
                image.load()
                # Close File Descriptor (or it'll reach OPEN_MAX)
                image_fp.close()
                self.data.append(image)
                self.label.append(class_idx)
            #print(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]
