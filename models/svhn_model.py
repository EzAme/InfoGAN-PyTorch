import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""
class VAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,1)
        self.bn1 =  nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8,16,3,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc_mu = nn.Sequential(nn.Linear(12544,3136),nn.ReLU(),nn.Linear(3136,124))
        self.fc_var = nn.Sequential(nn.Linear(12544,3136),nn.ReLU(),nn.Linear(3136,124))
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        mu = self.fc_mu(torch.flatten(x,start_dim=1))
        var =self.fc_var(torch.flatten(x,start_dim=1))
        return mu,var


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(168, 448, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(448)

        self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
    
    def forward(self, x):
        print("-"*25)
        print("Generator")
        print(x.shape)
        x = F.leaky_relu(self.bn1(self.tconv1(x)))
        print(x.shape)
        x = F.leaky_relu(self.bn2(self.tconv2(x)))
        print(x.shape)
        x = F.leaky_relu(self.tconv3(x))
        print(x.shape)
        x = F.leaky_relu(self.tconv4(x))
        print(x.shape)

        img = torch.tanh(self.tconv5(x))
        print(img.shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 40, 1)
        self.conv_mu = nn.Conv2d(128, 4, 1)
        self.conv_var = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
