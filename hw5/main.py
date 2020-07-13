import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import cv2
# my class
from model import Classifier
from dataset import ImgDataset
from utils import normalize, compute_saliency_maps, filter_explaination, smooth_grad


### READ IMAGE ###

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    if label:
        x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
        y = np.zeros((len(image_dir)), dtype=np.uint8)
    else:
        x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
        y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        if label:
            y[i] = int(file.split("_")[0])
            x[i, :, :] = cv2.resize(img,(128, 128))
        else:
            x[i, :, :] = cv2.resize(img,(128, 128))
    if label:
      return x, y
    else:
      return x

if __name__ == '__main__':
    #model_dir = sys.argv[1]
    workspace_dir = sys.argv[1]
    out_dir = sys.argv[2]
    model = Classifier().cuda()
    checkpoint = torch.load('VGG_150.pt')
    model.load_state_dict(checkpoint)

    train_paths, train_labels = readfile(workspace_dir, True)

    # 這邊在 initialize dataset 時只丟「路徑」和「class」，之後要從 dataset 取資料時
    # dataset 的 __getitem__ method 才會動態的去 load 每個路徑對應的圖片
    train_set = ImgDataset(train_paths, train_labels, mode='eval')

    # 指定想要一起 visualize 的圖片 indices
    img_indices = [83, 4218, 4707, 8598]
    img_indices2 = [993+200+709, 993+429+250+709]
    images, labels = train_set.getbatch(img_indices)
    images2, labels2 = train_set.getbatch2(img_indices)
    images3, labels3 = train_set.getbatch(img_indices2)

    saliencies = compute_saliency_maps(images, labels, model)
    
    # 使用 matplotlib 畫出來
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img)
        # 小知識：permute 是什麼，為什麼這邊要用?
        # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
        # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
        # 因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換
        # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其
        # - 第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height
        # - 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width
        # - 第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels

    plt.savefig('1.png')
    # 從第二張圖片的 saliency，我們可以發現 model 有認出蛋黃的位置
    # 從第三、四張圖片的 saliency，雖然不知道 model 細部用食物的哪個位置判斷，但可以發現 model 找出了食物的大致輪廓


    layer_activations = None
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=5, filterid=10, iteration=100, lr=0.1)
    fig, axs = plt.subplots(1, figsize=(15, 8))
    # 畫出 filter visualization
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(out_dir, '2.png'))
    # 根據圖片中的線條，可以猜測第 15 層 cnn 其第 0 個 filter 可能在認一些線條、甚至是 object boundary
    # 因此給 filter 看一堆對比強烈的線條，他會覺得有好多 boundary 可以 activate

    # 畫出 filter activations
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        axs[0][i].imshow(img)
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    plt.savefig(os.path.join(out_dir, '3.png'))


    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=5, filterid=20, iteration=100, lr=0.1)
    fig, axs = plt.subplots(1, figsize=(15, 8))
    # 畫出 filter visualization
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(out_dir, '4.png'))
    # 根據圖片中的線條，可以猜測第 15 層 cnn 其第 0 個 filter 可能在認一些線條、甚至是 object boundary
    # 因此給 filter 看一堆對比強烈的線條，他會覺得有好多 boundary 可以 activate

    # 畫出 filter activations
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        axs[0][i].imshow(img)
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    plt.savefig(os.path.join(out_dir, '5.png'))


    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=10, filterid=0, iteration=100, lr=0.1)
    fig, axs = plt.subplots(1, figsize=(15, 8))
    # 畫出 filter visualization
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(out_dir, '6.png'))
    # 根據圖片中的線條，可以猜測第 15 層 cnn 其第 0 個 filter 可能在認一些線條、甚至是 object boundary
    # 因此給 filter 看一堆對比強烈的線條，他會覺得有好多 boundary 可以 activate

    # 畫出 filter activations
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
        axs[0][i].imshow(img)
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    plt.savefig(os.path.join(out_dir, '7.png'))
    # 從下面四張圖可以看到，activate 的區域對應到一些物品的邊界，尤其是顏色對比較深的邊界
    
    def predict(input):
        # input: numpy array, (batches, height, width, channels)                                                                                                                                                     
        
        model.eval()                                                                                                                                                             
        input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                            
        # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
        # 也就是 (batches, channels, height, width)

        output = model(input.cuda())                                                                                                                                             
        return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                                
    def segmentation(input):
        # 利用 skimage 提供的 segmentation 將圖片分成 100 塊                                                                                                                                      
        return slic(input, n_segments=100, compactness=1, sigma=1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))                                                                                                                                                                 
    np.random.seed(16)
    # 讓實驗 reproducible
    for idx, (image, label) in enumerate(zip(images3.permute(0, 2, 3, 1).numpy(), labels3)):                                                                                                                                             
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array

        explainer = lime_image.LimeImageExplainer()                                                                                                                              
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
        # classifier_fn 定義圖片如何經過 model 得到 prediction
        # segmentation_fn 定義如何把圖片做 segmentation
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

        lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                    label=label.item(),                                                                                                                           
                                    positive_only=False,                                                                                                                         
                                    hide_rest=False,                                                                                                                             
                                    num_features=11,                                                                                                                              
                                    min_weight=0.05                                                                                                                              
                                )
        # 把 explainer 解釋的結果轉成圖片
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
        print(lime_img.shape)
        axs[idx].imshow(lime_img[:, :, [2,1,0]])

    plt.savefig(os.path.join(out_dir, '8.png'))

    fig, axs = plt.subplots(1, 4, figsize=(15, 8))                                                                                                                                                                 
    np.random.seed(16)
    # 讓實驗 reproducible
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):                                                                                                                                             
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array

        explainer = lime_image.LimeImageExplainer()                                                                                                                              
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
        # classifier_fn 定義圖片如何經過 model 得到 prediction
        # segmentation_fn 定義如何把圖片做 segmentation
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

        lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                    label=label.item(),                                                                                                                           
                                    positive_only=False,                                                                                                                         
                                    hide_rest=False,                                                                                                                             
                                    num_features=11,                                                                                                                              
                                    min_weight=0.05                                                                                                                              
                                )
        # 把 explainer 解釋的結果轉成圖片
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
        print(lime_img.shape)
        axs[idx].imshow(lime_img[:, :, [2,1,0]])

    plt.savefig(os.path.join(out_dir, '9.png'))


    times = 3
    saliencies = smooth_grad(images2, labels2, model, times)

    # 使用 matplotlib 畫出來
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img)
        # 小知識：permute 是什麼，為什麼這邊要用?
        # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
        # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
        # 因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換
        # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其
        # - 第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height
        # - 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width
        # - 第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels

    plt.savefig(os.path.join(out_dir, '10.png'))
    # 從第二張圖片的 saliency，我們可以發現 model 有認出蛋黃的位置
    # 從第三、四張圖片的 saliency，雖然不知道 model 細部用食物的哪個位置判斷，但可以發現 model 找出了食物的大致輪廓

    times = 10
    saliencies = smooth_grad(images2, labels2, model, times)

    # 使用 matplotlib 畫出來
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img)
        # 小知識：permute 是什麼，為什麼這邊要用?
        # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
        # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
        # 因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換
        # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其
        # - 第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height
        # - 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width
        # - 第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels

    plt.savefig(os.path.join(out_dir, '11.png'))
    # 從第二張圖片的 saliency，我們可以發現 model 有認出蛋黃的位置
    # 從第三、四張圖片的 saliency，雖然不知道 model 細部用食物的哪個位置判斷，但可以發現 model 找出了食物的大致輪廓


    times = 100
    saliencies = smooth_grad(images2, labels2, model, times)

    # 使用 matplotlib 畫出來
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            img = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img)
        # 小知識：permute 是什麼，為什麼這邊要用?
        # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
        # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
        # 因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換
        # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其
        # - 第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height
        # - 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width
        # - 第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels

    plt.savefig(os.path.join(out_dir, '12.png'))
    # 從第二張圖片的 saliency，我們可以發現 model 有認出蛋黃的位置
    # 從第三、四張圖片的 saliency，雖然不知道 model 細部用食物的哪個位置判斷，但可以發現 model 找出了食物的大致輪廓


    




