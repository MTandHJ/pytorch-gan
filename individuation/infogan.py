
import torch
import torch.nn.functional as F
from freegan.base import Coach, Generator



class InfoGANGenerator(Generator):

    def __init__(
        self, arch, device,
        dim_latent, criterion, 
        optimizer, learning_policy
    ):
        """
        criterion_n: for cls
        criterion_l: for labels
        criterion_c: for codes
        """
        super(InfoGANGenerator, self).__init__(
            arch, device,
            dim_latent, criterion, 
            optimizer, learning_policy
        )
        self.criterion_n, self.criterion_l, self.criterion_c = criterion
        self.dim_noise, self.dim_label, self.dim_code = dim_latent
        del self.criterion, self.dim_latent
        self._sampler_noise = torch.distributions.Uniform(0, 1).sample
        self._sampler_codes = torch.distributions.Uniform(-1, 1).sample
        self._sampler_labels = torch.randint

    def sample_noise(self, batch_size):
        return self._sampler_noise((batch_size, self.dim_noise)).to(self.device)

    def sample_labels(self, batch_size):
        temp = self._sampler_labels(0, self.dim_label, (batch_size,))
        return F.one_hot(temp, num_classes=self.dim_label).float().to(self.device)

    def sample_codes(self, batch_size):
        return self._sampler_codes((batch_size, self.dim_code)).to(self.device)

    
    def sampler(self, batch_size):
        noise = self.sample_noise(batch_size)
        labels = self.sample_labels(batch_size)
        codes = self.sample_codes(batch_size)
        return noise, labels, codes

    @torch.no_grad()
    def evaluate(self):
        noise = self.sample_noise(self.dim_label)
        labels = torch.arange(0, 10)
        labels = F.one_hot(labels, num_classes=self.dim_label).float().to(self.device)
        codes = torch.zeros((self.dim_label, self.dim_code)).to(self.device)
        inputs1 = torch.cat((noise, labels, codes), dim=1)
        imgs1 = self(inputs1)
        labels = torch.ones(self.dim_label) * torch.randint(0, self.dim_label, (1,))
        labels = F.one_hot(labels.long(), num_classes=self.dim_label).float().to(self.device)
        temp = torch.randint(0, self.dim_code, (1,)).item()
        codes[:, temp] = torch.linspace(-1, 1, self.dim_label).to(self.device)
        inputs2 = torch.cat((noise, labels, codes), dim=1)
        imgs2 = self(inputs2)
        return imgs1, imgs2


class InfoGANCoach(Coach):
    def __init__(
        self, generator,
        discriminator,
        device, normalizer,
        weight_cat=1., weight_con=0.1,
    ):
        """
        weight_cat: the weight for cls part loss
        weight_con: the weight for codes part loss
        """

        super(InfoGANCoach, self).__init__(
            generator, discriminator,
            device, normalizer
        )

        self.weight_cat = weight_cat
        self.weight_con = weight_con
        
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
            l1, l2, l3 = self.generator.sampler(batch_size)
            l = torch.cat((l1, l2, l3), dim=1)
            inputs_fake = self.generator(l)
            probs, _, _ = self.discriminator(self.normalizer(inputs_fake))
            loss_g = self.generator.criterion_n(probs, labels_real)

            # update the generator
            self.generator.optimizer.zero_grad()
            loss_g.backward()
            self.generator.optimizer.step()

            # discriminator part
            self.generator.eval()
            self.discriminator.train()
            inputs = torch.cat((inputs_real, inputs_fake.detach()), dim=0)
            probs, _, _ = self.discriminator(self.normalizer(inputs))
            labels = torch.cat((labels_real, labels_fake), dim=0)
            loss_d = self.discriminator.criterion(probs, labels)

            # update the discriminator
            self.discriminator.optimizer.zero_grad()
            loss_d.backward()
            self.discriminator.optimizer.step()

            # log
            validity = (probs.round() == labels).sum().item()
            self.loss_g.update(loss_g.item(), n=batch_size, mode="mean")
            self.loss_d.update(loss_d.item(), n=batch_size, mode="mean")
            self.validity.update(validity, n=batch_size * 2, mode="sum")

            # info part
            self.generator.train()
            self.discriminator.train()
            l1, l2, l3 = self.generator.sampler(batch_size)
            l = torch.cat((l1, l2, l3), dim=1)
            inputs_fake = self.generator(l)
            _, logits, codes = self.discriminator(self.normalizer(inputs_fake))
            loss1 = self.generator.criterion_l(logits, l2.argmax(-1))
            loss2 = self.generator.criterion_c(codes, l3)
            loss = loss1 * self.weight_cat + loss2 * self.weight_con            

            # update both
            self.generator.optimizer.zero_grad()
            self.discriminator.optimizer.zero_grad()
            loss.backward()
            self.generator.optimizer.step()
            self.discriminator.optimizer.step()

        
        self.progress.display(epoch=epoch)
        self.generator.learning_policy.step()
        self.discriminator.learning_policy.step()

        return self.loss_g.avg, self.loss_d.avg, self.validity.avg