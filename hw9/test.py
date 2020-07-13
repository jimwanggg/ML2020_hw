import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from preprocess import preprocess, Image_Dataset
from dim import cal_acc
from model import AE
import sys

out = sys.argv[3]
model_path = sys.argv[2]
npy_path = sys.argv[1]
def inference(X, model, batch_size=256):
    X, X2 = preprocess(X)
    dataset = Image_Dataset(X2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, (x, y) in enumerate(dataloader):
        #x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    
    #transformer = KernelPCA(n_components=350, kernel='rbf', n_jobs=-1, random_state=0)
    #kpca = transformer.fit_transform(latents)
    #print('First Reduction Shape:', kpca.shape)

    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1, random_state=0)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    transformer = KernelPCA(n_components=50, kernel='rbf', n_jobs=-1, random_state=0)
    kpca = transformer.fit_transform(kpca)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

# load model
model = AE().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# 準備 data
trainX = np.load(npy_path)

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

# 將預測結果存檔，上傳 kaggle
#save_prediction(pred, 'prediction.csv')

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
save_prediction(invert(pred), out)