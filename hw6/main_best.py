import os
import sys
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
import torch.nn as nn
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = img.convert('RGB')
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200


class Attacker:
    def __init__(self, img_dir, label):
        # 讀入預訓練模型 vgg16
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    # FGSM 攻擊
    def fgsm_attack(self, image, epsilon, data_grad):
        # 找出 gradient 的方向
        sign_data_grad = data_grad.sign()
        # 將圖片加上 gradient 方向乘上 epsilon 的 noise

        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        for (data, target) in self.loader:
            #print(data.size())
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                adv_ex = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), init_pred.item(), data_raw , adv_ex) )
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            loss = F.nll_loss(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            print(data_grad)
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
          
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )        
        final_acc = 1 - (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, success+wrong, len(self.loader), final_acc))
        return adv_examples, final_acc

    def attack_best(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        lists = [20, 40, 60]
        cnt = 0
        criterion = nn.CrossEntropyLoss()
        for (data, target) in self.loader:
            cnt += 1
            #print(data.size())
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            for epoch in range(30):
                output = self.model(data)
                init_pred = output.max(1, keepdim=True)[1]
                # 如果 class 錯誤 就離開
                if init_pred.item() != target.item():
                    break

                # 如果 class 一樣就開始計算 gradient 進行 FGSM 攻擊
                loss = criterion(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                data.data = self.fgsm_attack(data, epsilon, data_grad)
                #print('epoch', epoch, '...')
                data.grad.data.zero_()

            if init_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
            adv_ex = data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), init_pred.item(), data_raw , adv_ex) )        
        final_acc = 1 - (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, success+wrong, len(self.loader), final_acc))
        return adv_examples, final_acc

if __name__ == '__main__':
    # 讀入圖片相對應的 label
    store = sys.argv[2]
    input_file = sys.argv[1]
    df = pd.read_csv(os.path.join(input_file, "labels.csv"))
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(os.path.join(input_file, "categories.csv"))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    #attacker2 = Attacker('./store', df)
    attacker = Attacker(os.path.join(input_file, 'images'), df)
    df_list = df.tolist()
    # 要嘗試的 epsilon
    epsilons = [0.04]
    print(epsilons)
    accuracies, examples = [], []
    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        #ex, acc= attacker2.attack(eps)
        ex, acc = attacker.attack_best(eps)
        accuracies.append(acc)
        examples.append(ex)


    max_value = 0
    max_index = 0
    for index, acc in enumerate(epsilons):
        if acc >= max_value:
            max_value = acc
            max_index = index
    
    for i in range(len(examples[max_index])):
        #print(i)
        #plt.figure()
        a, b, org, image = examples[max_index][i]
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)
        #norm = np.linalg.norm(image - org, np.inf)
        #print('norm = ', norm)
        #plt.imshow(image)
        name = '%03d.png' % i
        #image.save(name)
        matplotlib.image.imsave(os.path.join(store, name), image)
        #plt.savefig(os.path.join(store, name))

'''
    cnt = 0
    plt.figure(figsize=(30, 30))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,orig_img, ex = examples[i][j]
            # plt.title("{} -> {}".format(orig, adv))
            plt.title("original: {}".format(label_name[orig].split(',')[0]))
            orig_img = np.transpose(orig_img, (1, 2, 0))
            plt.imshow(orig_img)
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()
'''