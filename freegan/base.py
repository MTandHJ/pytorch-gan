


import torch
import torch.nn as nn
from collections.abc import Iterable
from .utils import AverageMeter, ProgressMeter




class Generator(nn.Module):

    def __init__(
        self, arch, device,
        dim_latent, criterion, 
        optimizer, learning_policy
    ):
        super(Generator, self).__init__()

        if isinstance(dim_latent, Iterable):
            self.dim_latent = list(dim_latent)
        else:
            self.dim_latent = [dim_latent]

        self.arch = arch
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_policy = learning_policy
    
    def sampler(self, batch_size, rtype="gaussian"):
        size = [batch_size] + self.dim_latent
        if rtype == "gaussian":
            return torch.randn(size).to(self.device)
        elif rtype == "uniform":
            return torch.rand(size).to(self.device)
        else:
            raise NotImplementedError(f"No such rtype {rtype}.")
    
    @torch.no_grad()
    def evaluate(self, z=None, times=10):
        if z is None:
            z = self.sampler(times)
        else:
            z = z.to(self.device)
        self.arch.eval()
        imgs = self.arch(z)
        return imgs

    def save(self, path):
        torch.save(self.arch.state_dict(), path + "/generator_paras.pt")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super(Generator, self).state_dict(
            destination, prefix, keep_vars
        )
        destination['optimizer'] = self.optimizer.state_dict()
        destination['learning_policy'] = self.learning_policy.state_dict()

        return destination

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.learning_policy.load_state_dict(state_dict['learning_policy'])
        del state_dict['optimizer'], state_dict['learning_policy']
        return super(Discriminator, self).load_state_dict(state_dict, strict)

    def query(self, z):
        return self.arch(z)

    def forward(self, z):
        return self.query(z)


class Discriminator(nn.Module):

    def __init__(
        self, arch, device, criterion,
        optimizer, learning_policy
    ):
        super(Discriminator, self).__init__()

        self.arch = arch
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_policy = learning_policy

    def save(self, path):
        torch.save(self.arch.state_dict(), path + "/discriminator_paras.pt")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super(Discriminator, self).state_dict(
            destination, prefix, keep_vars
        )
        destination['optimizer'] = self.optimizer.state_dict()
        destination['learning_policy'] = self.learning_policy.state_dict()

        return destination

    def load_state_dict(self, state_dict, strict=True):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.learning_policy.load_state_dict(state_dict['learning_policy'])
        del state_dict['optimizer'], state_dict['learning_policy']
        return super(Discriminator, self).load_state_dict(state_dict, strict)

    def query(self, x):
        return self.arch(x)

    def forward(self, x):
        return self.query(x)


class Coach:

    def __init__(
        self, generator: Generator, 
        discriminator: Discriminator,
        device, normalizer, 
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.normalizer = normalizer
        self.loss_g = AverageMeter("Loss_G")
        self.loss_d = AverageMeter("Loss_D")
        self.validity = AverageMeter("Validity")
        self.progress = ProgressMeter([self.loss_g, self.loss_d, self.validity])

    def save(self, path):
        self.generator.save(path)
        self.discriminator.save(path)

    def train(self, trainloader, *, epoch=8888):
        self.progress.step() # reset the meter
        for inputs_real, _ in trainloader:
            batch_size = inputs_real.size(0)
            labels_real = torch.ones(batch_size).to(self.device)
            labels_fake = torch.zeros(batch_size).to(self.device)
            inputs_real = inputs_real.to(self.device)

            
            # generator part
            self.generator.train()
            self.discriminator.eval()
            z = self.generator.sampler(inputs_real.size(0))
            inputs_fake = self.generator(z)
            outs_g = self.discriminator(self.normalizer(inputs_fake))
            loss_g = self.generator.criterion(outs_g, labels_real) # real...

            # update the generator
            self.generator.optimizer.zero_grad()
            loss_g.backward()
            self.generator.optimizer.step()

            # discriminator part
            self.generator.eval()
            self.discriminator.train()
            inputs = torch.cat((inputs_real, inputs_fake.detach()), dim=0)
            labels = torch.cat((labels_real, labels_fake), dim=0)
            outs_d = self.discriminator(self.normalizer(inputs))
            loss_d = self.discriminator.criterion(outs_d, labels)

            # update the discriminator
            self.discriminator.optimizer.zero_grad()
            loss_d.backward()
            self.discriminator.optimizer.step()

            # log
            validity = (outs_d.round() == labels).sum().item()
            self.loss_g.update(loss_g.item(), n=batch_size, mode="mean")
            self.loss_d.update(loss_d.item(), n=batch_size, mode="mean")
            self.validity.update(validity, n=batch_size * 2, mode="sum")
        
        self.progress.display(epoch=epoch)
        self.generator.learning_policy.step()
        self.discriminator.learning_policy.step()

        return self.loss_g.avg, self.loss_d.avg, self.validity.avg













    





