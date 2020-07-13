import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import time
import random
import numpy as np
from data import MyDataset, AllDataset
from model import StudentNet, MobileNetv2, TANet

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def loss_fn_kd2(outputs, another_outputs, labels, teacher_outputs, T=20, alpha=0.5, beta = 0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha - beta)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    if teacher_outputs is None:
        soft_loss1 = 0
    else:
        soft_loss1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    soft_loss2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(another_outputs/T, dim=1)) * (beta * T * T)
    return hard_loss + soft_loss1 + soft_loss2

trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'../hw3/food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform,
        label = False if mode == 'testing' else True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training')
    )

    return dataset, dataloader

def get_dataloader_ALL(modes, batch_size=32):
    
    foldernames = []
    for name in modes:
        foldernames.append(f'../hw3/food-11/{name}')
    dataset = AllDataset(
        foldernames,
        transform=trainTransform,
        label = True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = True,
    )

    return dataset, dataloader

# get dataloader
train_dataset, train_dataloader = get_dataloader('training', batch_size=32)
valid_dataset, valid_dataloader = get_dataloader('validation', batch_size=32)
#all_dataset, all_dataloader = get_dataloader_ALL(['training', 'validation'], batch_size=32)

teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
#student_net = TANet(base=16).cuda()
student_net = StudentNet().cuda()
### mutual learning ###
#student_net2 = StudentNet().cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18_from_scratch.bin'))
optimizer = optim.AdamW(student_net.parameters(), lr=3e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max') # adjust lr
optimizer2 = optim.AdamW(teacher_net.parameters(), lr=5e-4)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'max') # adjust lr

def run_epoch(net, dataloader, update=True, alpha=0.1, T = 50):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, T, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, T, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

def run_epoch_mutual(net1, net2, dataloader, update=True, alpha=0, beta = 0.5, T = 50):
    total_num, total_hit, total_loss, total_hit2, total_loss2 = 0, 0, 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        #with torch.no_grad():
        #    soft_labels = teacher_net(inputs)

        if update:
            logits = net1(inputs)
            logits2 = net2(inputs)
            
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd2(logits, logits2, hard_labels, None, T, alpha, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logits = net1(inputs)
            logits2 = net2(inputs)
            loss2 = loss_fn_kd2(logits2, logits, hard_labels, None, T, alpha, 0.1)

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()      
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = net1(inputs)
                logits2 = net2(inputs)
                loss = loss_fn_kd2(logits, logits2, hard_labels, None, T, alpha, beta)
                loss2 = loss_fn_kd2(logits2, logits, hard_labels, None, T, alpha, 0.1)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_hit2 += torch.sum(torch.argmax(logits2, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
        total_loss2 += loss2.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num, total_loss2 / total_num, total_hit2 / total_num


# TeacherNet永遠都是Eval mode.
#teacher_net.eval()

now_best_acc = 0
now_best_acc2 = 0
for epoch in range(200):
    now = time.time()
    student_net.train()
    #student_net2.train()
    teacher_net.train()
    train_loss, train_acc, train_loss2, train_acc2 = run_epoch_mutual(student_net, teacher_net, train_dataloader, update=True)
    student_net.eval()
    #student_net2.eval()
    teacher_net.eval()
    valid_loss, valid_acc, valid_loss2, valid_acc2 = run_epoch_mutual(student_net, teacher_net, valid_dataloader, update=False)
    scheduler.step(valid_acc) # one epoch one step
    scheduler2.step(valid_acc2) # one epoch one step
    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model_scratch.bin')
    #if valid_acc2 > now_best_acc:
    #    now_best_acc2 = valid_acc2
    #    torch.save(teacher_net.state_dict(), 'teacher_model_2.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss2, train_acc2, valid_loss2, valid_acc2))
    print('time : {:3.2f}'.format(time.time() - now))

'''
now_best_acc = 0
now_best_acc2 = 0

student_net_all = StudentNet().cuda()
#student_net_all2 = StudentNet().cuda()
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
optimizer = optim.AdamW(student_net_all.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # adjust lr
optimizer2 = optim.AdamW(teacher_net.parameters(), lr=5e-4)
scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min') # adjust lr

for epoch in range(250):
    now = time.time()
    student_net_all.train()
    #student_net_all2.train()
    teacher_net.train()
    train_loss, train_acc, train_loss2, train_acc2 = run_epoch_mutual(student_net_all, teacher_net, all_dataloader, update=True)
    student_net_all.eval()
    #student_net_all2.eval()
    teacher_net.eval()
    scheduler.step(train_loss) # one epoch one step
    scheduler2.step(train_loss2)
    # 存下最好的model。
    if train_acc > now_best_acc:
        now_best_acc = train_acc
        torch.save(student_net_all.state_dict(), 'student_model_mutual_1.bin')
    if epoch % 10 == 9:
        torch.save(student_net_all.state_dict(), 'student_model_mutual_teacher_{}.bin'.format(epoch+1))
    #if train_acc2 > now_best_acc2:
    #    now_best_acc = train_acc2
    #    torch.save(student_net_all2.state_dict(), 'student_model_all_2.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} '.format(
        epoch, train_loss, train_acc))
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} '.format(
        epoch, train_loss2, train_acc2))
    print('time : {:3.2f}'.format(time.time() - now))
'''

