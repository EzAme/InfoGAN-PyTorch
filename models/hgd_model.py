import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""
class VAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv_mu = nn.Conv2d(128, 228, 1)
        self.conv_var = nn.Conv2d(128, 228, 1)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        mu = self.conv_mu(x)
        var = self.conv_var(x)
        return mu,var


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(336, 512, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        self.tconv5 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False)

        self.tconv6 = nn.ConvTranspose2d(32, 1, 4, 2, padding=1, bias=False)
    
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.tconv1(x)))
        x = F.leaky_relu(self.bn2(self.tconv2(x)))
        x = F.leaky_relu(self.tconv3(x))
        x = F.leaky_relu(self.tconv4(x))

        img = torch.tanh(self.tconv6(self.tconv5(x)))
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.1, inplace=True)

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

        self.conv_disc = nn.Conv2d(128, 100, 1)
        self.conv_mu = nn.Conv2d(128, 8, 1)
        self.conv_var = nn.Conv2d(128, 8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
