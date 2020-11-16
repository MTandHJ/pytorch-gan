

import torch
import torch.nn as nn
from freegan.base import Coach
from freegan.utils import AverageMeter, ProgressMeter




class LAPCoach:

    def __init__(
        self, generators,
        discriminators,
        device
    ):
        assert len(generators) == len(discriminators), \
                "The number of generators and discriminators should be the same."
        self.generators = generators
        self.discriminators = discriminators
        self.device = device
        self.opd = nn.MaxPool2d(2, stride=2, padding=0)
        self.opu = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()
        self.levels = len(generators)

        self.loss_g = AverageMeter("Loss_G")
        self.loss_d = AverageMeter("Loss_D")
        self.validity = AverageMeter("Validity")
        self.progress = ProgressMeter([self.loss_g, self.loss_d, self.validity])
        
    def save(self, path):
        for level in range(self.levels):
            self.generators[level].save(path, postfix=str(level))
            self.discriminators[level].save(path, postfix=str(level))

    def step(self):
        for level in range(self.levels):
            generator = self.generators[level]
            discriminator = self.discriminators[level]
            generator.learning_policy.step()
            discriminator.learning_policy.step()

    @torch.no_grad()
    def evaluate(self, batch_size=10):
        z = self.generators[-1].sampler(batch_size, rtype="uniform")
        x = self.generators[-1](z)
        for level in range(2, self.levels+1):
            generator = self.generators[-level]
            x = self.opu(x) # upsample
            z = generator.sampler(batch_size, rtype="uniform")
            temp = torch.cat((x, z), dim=1)
            h = generator(temp)
            x = self.sigmoid(x + h)
        return x

    def ahead(self, imgs, level):
        downsamples = self.opd(imgs)
        upsamples = self.opu(downsamples)
        diffs = imgs - upsamples
        generator = self.generators[level]
        discriminator = self.discriminators[level]

        batch_size = imgs.size(0)
        labels_real = torch.ones(batch_size).to(self.device)
        labels_fake = torch.zeros(batch_size).to(self.device)
        
        # generator part
        generator.train()
        discriminator.eval()
        z = generator.sampler(batch_size, rtype="uniform")
        if level < self.levels-1:
            temp = torch.cat((upsamples, z), dim=1)
            inputs_fake = generator(temp) + upsamples
        else:
            inputs_fake = generator(z)
        outs_g = discriminator(inputs_fake)
        loss_g = generator.criterion(outs_g, labels_real) # real...

        # update the generator
        generator.optimizer.zero_grad()
        loss_g.backward()
        generator.optimizer.step()

        # discriminator part
        generator.eval()
        discriminator.train()
        inputs = torch.cat((imgs, inputs_fake.detach()), dim=0)
        labels = torch.cat((labels_real, labels_fake), dim=0)
        outs_d = discriminator(inputs)
        loss_d = discriminator.criterion(outs_d, labels)

        # update the discriminator
        discriminator.optimizer.zero_grad()
        loss_d.backward()
        discriminator.optimizer.step()

        validity = (outs_d.round() == labels).sum().item()
        self.loss_g.update(loss_g.item(), n=batch_size, mode="mean")
        self.loss_d.update(loss_d.item(), n=batch_size, mode="mean")
        self.validity.update(validity, n=batch_size * 2, mode="sum")

        return self.sigmoid(downsamples).detach()

    def train(self, trainloader, *, epoch=8888):
        self.progress.step() # reset the meter
        for inputs, _ in trainloader:
            inputs = inputs.to(self.device)
            for level in range(self.levels):
                inputs = self.ahead(inputs, level)
        
        self.progress.display(epoch=epoch)
        self.step()

        return self.loss_g.avg, self.loss_d.avg, self.validity.avg
    
