

import torch
import torch.nn as nn
from .utils import down



class Generator(nn.Module):

    def __init__(self, out_shape, dim_input=128):
        super(Generator, self).__init__()

        c = out_shape[0]
        self.h = out_shape[1] // 4
        self.w = out_shape[2] // 4
        self.fc = nn.Linear(dim_input, 128 * self.h * self.w)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128), # n x 128 x h x w
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2), # n x 128 x 2h x 2w
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # n x 128 x 2h x 2w
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True), 
            nn.Upsample(scale_factor=2), # n x 128 x 4h x 4w
            nn.Conv2d(128, 64, 3, stride=1, padding=1), # n x 64 x 4h x 4w
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, c, 3, stride=1, padding=1), # n x c x 4h x 4w
            # no batchnorm at the output layer as the paper said
            nn.Sigmoid() # not tanh as used in the paper
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0., 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1., 0.02)
                nn.init.constant_(m.bias.data, 0.)

    def forward(self, inputs):
        l1 = self.fc(inputs).view(inputs.size(0), -1, self.h, self.w)
        outs = self.conv(l1)
        return outs

class Discriminator(nn.Module):

    def __init__(self, in_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True), # use leakyrelu instead of relu
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv = nn.Sequential(
            # no batchnorm at the input latyer as the paper suggested
            *discriminator_block(in_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        h = down(in_shape[1], 3, 4, 2, 1)
        w = down(in_shape[1], 3, 4, 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(128 * h * w, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0., 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1., 0.02)
                nn.init.constant_(m.bias.data, 0.)

    def forward(self, inputs):
        features = self.conv(inputs).flatten(start_dim=1)
        probs = self.fc(features)
        return probs.squeeze()



if __name__ == "__main__":

    g = Generator((1, 28, 28))
    d = Discriminator((1, 28, 28))
    z = torch.rand(10, 128)    
    x = g(z)
    print(x.size())
    y = d(x)
    print(y.size())










