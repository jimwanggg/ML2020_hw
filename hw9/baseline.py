from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from func import same_seeds
import torchvision.transforms as transforms
from preprocess import preprocess, Image_Dataset2
from model import AE
import sys

npy_path = sys.argv[1]
model_path = sys.argv[2]

trainX = np.load(npy_path)

trainX_preprocessed, trainX_preprocessed2 = preprocess(trainX)
img_dataset = Image_Dataset2(trainX_preprocessed)

same_seeds(0)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # adjust lr

model.train()
n_epoch = 300


# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)


# 主要的訓練過程
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (epoch+1) % 10 == 0:
        #    torch.save(model.state_dict(), './checkpoints/checkpoint_{:03d}.pth'.format(epoch+1))
    scheduler.step(loss.data)
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

# 訓練完成後儲存 model
torch.save(model.state_dict(), model_path)
