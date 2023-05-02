import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Architecture based on InfoGAN paper.
"""

class VAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.label = nn.Linear(1,4096) # reshape to batchx1X64x64
        # self.label = nn.Linear(1,64,bias=False) # reshape to batchx1x8x8
        self.Elabel = nn.Embedding(10,1) # reshape to batchx1x8x8
        self.conv1 = nn.Conv2d(2, 128, 4, 2, 1)
        # self.conv1_1 = nn.Conv2d(2, 128, 3, 2, 1)
        self.conv2 = nn.Conv2d(128, 512, 4, 2, 1,bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 256, 4, 2, 1,bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 8,bias = False)
        self.bn4 = nn.BatchNorm2d(256)

  

        self.conv_mu = nn.Conv2d(256, 256, 1)
        self.conv_var = nn.Conv2d(256, 256, 1)
        self.relu = nn.LeakyReLU()
    def forward(self,x,label):
        x = torch.cat((x,self.label(self.Elabel(label)).view(-1,1,64,64)),1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        mu = self.conv_mu(x)
        var = torch.exp(self.conv_var(x))
        epsilon = Variable(torch.randn_like(var.detach()))
        z = mu + epsilon*torch.sqrt(var)
        return z.view(-1,256), mu, var
    

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_latent = nn.Linear(256,16384,bias=False) #reshape batch,128x8*8
        self.label = nn.Linear(1,64,bias=False) # reshape to batchx1x8x8
        self.Elabel = nn.Embedding(10,1) # reshape to batchx1x8x8
        self.cont_latent = nn.Linear(4,256,bias=False) # reshape to batchx4x8x8
        self.tconv1 = nn.ConvTranspose2d(256+1+4, 256, 4,2,1,bias=False)
    
        self.bn1 = nn.BatchNorm2d(256+1+4)
        # self.tconv2 = nn.PixelShuffle(2)
        
        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.tconv3 = nn.ConvTranspose2d(128, 1, 4, 2, padding=1,bias=False)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x,label,cont):
        
        x = self.bn1(torch.cat((F.relu(self.noise_latent(x)).view(-1,256,8,8),self.label(self.Elabel(label)).view(-1,1,8,8),self.cont_latent(cont).view(-1,4,8,8)),1))
        # x = self.up1(x)
        x = F.relu(self.bn2(self.tconv1(x)))
        # x = self.up2(x)
        x = F.relu((self.tconv2(x)))
        # x = self.up3(x)
        img = torch.tanh((self.tconv3(x)))
        return img


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(142, 128, 3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 5,1,2),  
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, 5,1,2),
        )
    def forward(self,x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.noise_latent = nn.Linear(128,8192) #reshape batch,128x8*8
        self.label = nn.Linear(1,4096) # reshape to batchx1X64x64
        # self.label = nn.Linear(1,64,bias=False) # reshape to batchx1x8x8
        self.Elabel = nn.Embedding(10,1) # reshape to batchx1x8x8
        self.conv1 = nn.Conv2d(2, 128, 4, 2, 1)
        # self.conv1_1 = nn.Conv2d(2, 128, 3, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1,bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 142, 4, 2, 1,bias=False)
        self.bn3 = nn.BatchNorm2d(142)
        self.leaky_relu1 = nn.LeakyReLU(0.1,inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(0.1,inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(0.1,inplace=True)
    def forward(self, x,label):
        x = torch.cat((x,self.label(self.Elabel(label)).view(-1,1,64,64)),1)
        x = self.leaky_relu1(self.conv1(x))
        # x = F.leaky_relu(self.conv1_1(x), 0.1, inplace=True)

        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.leaky_relu3(self.bn3(self.conv3(x)))

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(142, 1, 8)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class LogitHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(142, 256, 5,2,2)
        self.fc1 = nn.Linear(4096,256)
        self.fc2 = nn.Linear(256,10)
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        x = torch.flatten(x,start_dim = 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(142, 128, 8,bias = False)
        
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 4, 1)
        self.conv_var = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = F.softmax(self.conv_disc(x).squeeze())

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
