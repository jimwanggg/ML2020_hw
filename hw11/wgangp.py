import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt
from model import Generator, Discriminator_GP
from utils import same_seeds
from data import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import sys


if __name__ == "__main__":

    workspace_dir = sys.argv[1]
    checkpoint = sys.argv[2]

    # hyperparameters 
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 30
    #save_dir = os.path.join(workspace_dir, 'logs')
    #os.makedirs(save_dir, exist_ok=True)
    # Loss weight for gradient penalty
    lambda_gp = 10

    # model
    G = Generator(in_dim=z_dim).cuda()
    D = Discriminator_GP(3).cuda()
    G.train()
    D.train()

    # loss criterion
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


    same_seeds(0)
    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    dataset = get_dataset(workspace_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 隨便印一張圖看看

    plt.imshow(dataset[10].numpy().transpose(1,2,0))

    # for logging
    z_sample = Variable(torch.randn(100, z_dim)).cuda()

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        #interpolates = interpolates.type(torch.cuda.FloatTensor)
        d_interpolates = D(interpolates)
        d_interpolates = torch.unsqueeze(d_interpolates, 1)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        #print(real_samples.shape, fake_samples.shape, interpolates.shape, d_interpolates.shape)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)
            for _ in range(5):
                """ Train D """
                z = Variable(torch.randn(bs, z_dim)).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = G(z)

                # label        
                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()

                # dis
                r_logit = D(r_imgs.detach())
                f_logit = D(f_imgs.detach())
                
                # compute loss
                '''
                r_loss = criterion(r_logit, r_label)
                f_loss = criterion(f_logit, f_label)
                loss_D = (r_loss + f_loss) / 2
                '''
                penalty = compute_gradient_penalty(D, r_imgs.data, f_imgs.data)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + lambda_gp * penalty

                # update model
                D.zero_grad()
                loss_D.backward()
                opt_D.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)
            
            # compute loss
            #loss_G = criterion(f_logit, r_label)
            loss_G = -torch.mean(f_logit)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
        #G.eval()
        #f_imgs_sample = (G(z_sample).data + 1) / 2.0
        #filename = os.path.join(save_dir, f'WGAN-GP_Epoch_{epoch+1:03d}.jpg')
        #torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        #print(f' | Save some samples to {filename}.')
        # show generated image
        #grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        #plt.figure(figsize=(10,10))
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show()
        #G.train()
        #if (e+1) % 2 == 0:
    torch.save(G.state_dict(), checkpoint)
    #torch.save(D.state_dict(), os.path.join(workspace_dir, f'wganGP_d_{epoch+1}.pth'))