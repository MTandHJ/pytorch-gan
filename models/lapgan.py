


import torch
import torch.nn as nn



class Generator(nn.Module):

    def __init__(self, in_channels=3, middle_channels=64, out_channels=3):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+1, middle_channels, 3, 1, 1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, 3, 1, 1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(middle_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_channels * 2, out_channels, 3, 1, 1),
            # nn.Sigmoid()
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
            nn.Conv2d(middle_channels, middle_channels, 3, 2, 1),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(middle_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(middle_channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).flatten(start_dim=1)
        probs = self.fc(x)
        return probs.squeeze()


if __name__ == "__main__":

    opd = nn.MaxPool2d(2, stride=2, padding=0)
    opu = nn.Upsample(scale_factor=2)
    g1 = Generator(0, 8, 3)
    g2 = Generator(3, 16, 3)
    g3 = Generator(3, 32, 3)
    d1 = Discriminator(3, 8)
    d2 = Discriminator(3, 16)
    d3 = Discriminator(3, 32)

    z1 = torch.rand(5, 1, 8, 8)
    z2 = torch.rand(5, 1, 16, 16)
    z3 = torch.rand(5, 1, 32, 32)
    x1 = opu(g1(z1))
    x2 = opu(g2(torch.cat((x1, z2), dim=1)))
    x3 = g3(torch.cat((x2, z3), dim=1))
    print(x3.size())

    x3 = d3(x3)
    x2 = d2(x2)
    x1 = d1(x1)
















