#!/usr/bin/env python

"""
Goodfellow I., Pouget-Abadie J., Mirza M., Xu B., Warde-Farley D., Ozair S., Courville A. & Bengio Yoshua. 
Generative adversarial nets. ICLR, 2014.
"""


import argparse
from freegan.loadopts import *

METHOD = "LAPGAN"
VALID_EPOCHS = 20
FMT = "{description}=" \
        "={criterion_g}-{learning_policy_g}-{optimizer_g}-{lr_g}" \
        "={criterion_d}-{learning_policy_d}-{optimizer_d}-{lr_d}" \
        "={epochs}-{batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("-g", "--generator", type=str, default="lapgan-g")
parser.add_argument("-d", "--discriminator", type=str, default="lapgan-d")

# for generator
parser.add_argument("-cg", "--criterion_g", type=str, default="bce")
parser.add_argument("-og", "--optimizer_g", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrg", "--lr_g", "--LR_G", "--learning_rate_g", type=float, default=0.002)
parser.add_argument("-lpg", "--learning_policy_g", type=str, default="cosine", 
                help="learning rate schedule defined in config.py")

# for discriminator
parser.add_argument("-cd", "--criterion_d", type=str, default="bce")
parser.add_argument("-od", "--optimizer_d", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrd", "--lr_d", "--LR_D", "--learning_rate_d", type=float, default=0.002)
parser.add_argument("-lpd", "--learning_policy_d", type=str, default="none", 
                help="learning rate schedule defined in config.py")

# basic settings
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=0.,
                help="weight decay")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied in training mode.")
parser.add_argument("--is_load_checkpoint", action="store_true", default=False)
parser.add_argument("-m", "--description", type=str, default="train")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)

if opts.dataset in ("cifar10", "cifar100"):
    opts.dim_latent = [(1, 32, 32), (1, 16, 16), (1, 8, 8)]
    opts.channels = 3
    opts.middle_channels = (64, 32, 16)

elif opts.dataset in ("mnist", ):
    opts.dim_latent = [(1, 28, 28), (1, 14, 14), (1, 7, 7)]
    opts.channels = 1
    opts.middle_channels = (32, 16, 8)


def load_cfg():
    from individuation.lapgan import LAPCoach
    from freegan.dict2obj import Config
    from freegan.base import Generator, Discriminator
    from freegan.utils import gpu

    cfg = Config()

    # load model
    arch_g = load_model(model_type=opts.generator)
    arch_d = load_model(model_type=opts.discriminator)

    generators = [
        arch_g(
            in_channels=opts.channels,
            middle_channels=opts.middle_channels[k],
            out_channels=opts.channels
        ) for k in range(len(opts.dim_latent)-1)
    ]
    generators.append(arch_g(0, opts.middle_channels[-1], opts.channels))

    discriminators = [
        arch_d(
            in_channels=opts.channels,
            middle_channels=opts.middle_channels[k],
        ) for k in range(len(opts.dim_latent))
    ]

    device = gpu(*generators, *discriminators)

    # load dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        train=True
    )
    normalizer = load_normalizer(dataset_type=opts.dataset)

    # load optimizer and correspoding learning policy
    optimizer_g = [
        load_optimizer(
            generator, opts.optimizer_g, lr=opts.lr_g,
            momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
            weight_decay=opts.weight_decay
        ) for generator in generators
    ]
    optimizer_d = [
        load_optimizer(
            discriminator, opts.optimizer_d, lr=opts.lr_d,
            momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
            weight_decay=opts.weight_decay
        ) for discriminator in discriminators
    ]

    learning_policy_g = [
        load_learning_policy(
            optimizer, opts.learning_policy_g,
            T_max=opts.epochs
        ) for optimizer in optimizer_g
    ]
    learning_policy_d = [
        load_learning_policy(
            optimizer, opts.learning_policy_d,
            T_max=opts.epochs
        ) for optimizer in optimizer_d
    ]

    # load criteria
    criterion_g = load_loss_func(loss_type=opts.criterion_g)
    criterion_d = load_loss_func(loss_type=opts.criterion_d)

    # load generator
    all_generators = [
        Generator(
            arch=generators[k], device=device,
            dim_latent=opts.dim_latent[k], 
            criterion=criterion_g,
            optimizer=optimizer_g[k],
            learning_policy=learning_policy_g[k]
        ) for k in range(len(generators))
    ]
    all_discriminators = [
        Discriminator(
            arch=discriminators[k], device=device,
            criterion=criterion_d, 
            optimizer=optimizer_d[k],
            learning_policy=learning_policy_d[k]
        ) for k in range(len(discriminators))
    ]

    # generate the path for logging information and saving parameters
    cfg['info_path'], log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        generator=opts.generator,
        discriminator=opts.discriminator,
        description=opts.description
    )
    
    if opts.is_load_checkpoint:
        raise NotImplementedError("Sorry, no checkpoint used in LAPGAN.")
    else:
        cfg['start_epoch'] = 0


    # load coach
    cfg['coach'] = LAPCoach(
        generators=all_generators,
        discriminators=all_discriminators,
        device=device,
    )

    return cfg, log_path



def main(
    coach, trainloader,
    start_epoch, info_path
):
    from freegan.utils import save_checkpoint, imagemeter
    for epoch in range(start_epoch, opts.epochs+1):
        if epoch % VALID_EPOCHS == 0:

            imgs = coach.evaluate(batch_size=10)
            fp = imagemeter(imgs)
            writter.add_figure(f"Image-Epoch:{epoch}", fp, global_step=epoch)

        loss_g, loss_d, validity = coach.train(trainloader, epoch=epoch)
        writter.add_scalars("Loss", {"generator":loss_g, "discriminator":loss_d}, epoch)
        writter.add_scalar("Validity", validity, epoch)



if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from freegan.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(cfg.info_path, log_path)
    readme(cfg.info_path, opts)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()


    






    
