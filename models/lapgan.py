


import torch
import torch.nn as nn



class H(nn.Module):

    def __init__(self):
        super(H, self).__init__()

        self.opd = nn.MaxPool2d(2, stride=2, padding=0)
        self.opu = nn.Upsample(2)

    def forward(self, x_k):
        x = self.opd(x_k)
        h = self.opu(x)
        return x, h


class Generator(nn.Module):

    def __init__(self, in_channels=3, middle_channels=64, out_channels=3):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels+1, middle_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, 2, 2, 1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, 3, 2, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x + z
        # suppose the noise exists as an additional channel
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

        
        
class Discriminator(nn.Module):

    def __init__(self, in_channels=3, middle_channels=64):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, 2, 1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, 3, 2, 1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(middle_channels * 2)
        self.fc = nn.Linear(middle_channels * 2, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.pool(x)
        probs = self.fc(x)
        return probs



        















