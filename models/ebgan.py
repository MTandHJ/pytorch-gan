



import torch
import torch.nn as nn



class Generator(nn.Module):

    def __init__(self, out_shape, dim_input=128):
        super(Generator, self).__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.out_shape = torch.tensor(out_shape)
        self.dense = nn.Sequential(
            *block(dim_input, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.out_shape.prod())),
            nn.Sigmoid() # limiting the outputs in [0, 1]
        )

    def forward(self, z):
        img = self.dense(z)
        img = img.view(img.size(0), *self.out_shape)
        return img


class Discriminator(nn.Module): 

    def __init__(self, in_shape):
        super(Discriminator, self).__init__()

        self.in_shape = torch.tensor(in_shape)
        self.dense = nn.Sequential(
            nn.Linear(self.in_shape.prod().long(), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, inputs):
        inputs_ = inputs.flatten(start_dim=1)
        outs = self.dense(inputs_)
        return outs.squeeze().abs()


if __name__ == "__main__":
    
    generator = Generator((28, 28))
    discriminator = Discriminator((28, 28))

    z = torch.randn(10, 128)
    imgs = generator(z)
    print(imgs.size())
    probs = discriminator(imgs)
    print(probs)