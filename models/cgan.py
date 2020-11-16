

import torch
import torch.nn as nn





import torch
import torch.nn as nn



class Generator(nn.Module):

    def __init__(self, out_shape, dim_input=128, num_classes=10):
        super(Generator, self).__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.out_shape = torch.tensor(out_shape)
        self.dense = nn.Sequential(
            *block(dim_input + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.out_shape.prod())),
            nn.Sigmoid() # limiting the outputs in [0, 1]
        )

    def forward(self, z, y):
        y_ = self.label_emb(y)
        img = self.dense(torch.cat((z, y_), dim=1))
        img = img.view(img.size(0), *self.out_shape)
        return img


class Discriminator(nn.Module): 

    def __init__(self, in_shape, num_classes=10):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.in_shape = torch.tensor(in_shape)
        self.dense = nn.Sequential(
            nn.Linear(self.in_shape.prod().long() + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, y):
        y_ = self.label_emb(y)
        inputs_ = inputs.flatten(start_dim=1)
        outs = self.dense(torch.cat((inputs_, y_), dim=1))
        return outs.squeeze()


if __name__ == "__main__":
    
    generator = Generator((28, 28))
    discriminator = Discriminator((28, 28))

    z = torch.randn(10, 128)
    y = torch.randint(0, 10, (10,))
    imgs = generator(z, y)
    print(imgs.size())
    probs = discriminator(imgs, y)
    print(probs)