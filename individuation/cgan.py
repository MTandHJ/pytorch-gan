
import torch
from freegan.base import Coach, Generator



class CGANGenerator(Generator):

    @torch.no_grad()
    def evaluate(self, num_classes):
        z = self.sampler(num_classes)
        y = torch.arange(0, num_classes, dtype=torch.long, device=self.device)
        self.arch.eval()
        imgs = self.arch(z, y)
        return imgs


class CGANCoach(Coach):
    def __init__(
        self, generator,
        discriminator,
        device, normalizer, 
        num_classes=10
    ):
        super(CGANCoach, self).__init__(
            generator, discriminator,
            device, normalizer
        )
        self.num_classes = num_classes
        
    def train(self, trainloader, *, epoch=8888):
        self.progress.step() # reset the meter
        for inputs_real, y_real in trainloader:
            batch_size = inputs_real.size(0)
            labels_real = torch.ones(batch_size).to(self.device)
            labels_fake = torch.zeros(batch_size).to(self.device)
            inputs_real = inputs_real.to(self.device)
            y_real = y_real.to(self.device)

            
            # generator part
            self.generator.train()
            self.discriminator.eval()
            z = self.generator.sampler(batch_size)
            y_fake = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            inputs_fake = self.generator(z, y_fake)
            outs_g = self.discriminator(self.normalizer(inputs_fake), y_fake)
            loss_g = self.generator.criterion(outs_g, labels_real) # real...

            # update the generator
            self.generator.optimizer.zero_grad()
            loss_g.backward()
            self.generator.optimizer.step()

            # discriminator part
            self.generator.eval()
            self.discriminator.train()
            inputs = torch.cat((inputs_real, inputs_fake.detach()), dim=0)
            y = torch.cat((y_real, y_fake), dim=0)
            outs_d = self.discriminator(self.normalizer(inputs), y)
            labels = torch.cat((labels_real, labels_fake), dim=0)
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