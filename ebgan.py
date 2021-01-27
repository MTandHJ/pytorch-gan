#!/usr/bin/env python

"""
Zhao J., Mathieu M. & LeCun Y. Energy-based generative adversarial networks. ICLR, 2017.
"""


import argparse
from freegan.loadopts import *

METHOD = "EBGAN"
VALID_EPOCHS = 20
FMT = "{description}=" \
        "={dim_latent}-{criterion_g}-{learning_policy_g}-{optimizer_g}-{lr_g}" \
        "={criterion_d}-{learning_policy_d}-{optimizer_d}-{lr_d}" \
        "={epochs}-{batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("-g", "--generator", type=str, default="ebgan-g")
parser.add_argument("-d", "--discriminator", type=str, default="ebgan-d")

# for EBGAN
parser.add_argument("--margin", type=float, default=10.)

# for generator
parser.add_argument("--dim_latent", type=int, default=128)
parser.add_argument("-cg", "--criterion_g", type=str, default="energy")
parser.add_argument("-og", "--optimizer_g", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrg", "--lr_g", "--LR_G", "--learning_rate_g", type=float, default=0.002)
parser.add_argument("-lpg", "--learning_policy_g", type=str, default="cosine", 
                help="learning rate schedule defined in config.py")

# for discriminator
parser.add_argument("-cd", "--criterion_d", type=str, default="energy")
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



def load_cfg():
    from freegan.dict2obj import Config
    from freegan.base import Generator, Discriminator, Coach
    from freegan.utils import gpu, load_checkpoint

    cfg = Config()

    # load model
    arch_g = load_model(model_type=opts.generator)(
        out_shape=get_shape(opts.dataset),
        dim_input=opts.dim_latent
    )
    arch_d = load_model(model_type=opts.discriminator)(
        in_shape=get_shape(opts.dataset)
    )
    device = gpu(arch_g, arch_d)

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
    optimizer_g = load_optimizer(
        arch_g, opts.optimizer_g, lr=opts.lr_g,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    optimizer_d = load_optimizer(
        arch_d, opts.optimizer_d, lr=opts.lr_d,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy_g = load_learning_policy(
        optimizer_g, opts.learning_policy_g,
        T_max=opts.epochs
    )
    learning_policy_d = load_learning_policy(
        optimizer_d, opts.learning_policy_d,
        T_max=opts.epochs
    )

    # load criteria
    criterion_g = load_loss_func(loss_type=opts.criterion_g, margin=opts.margin)
    criterion_d = load_loss_func(loss_type=opts.criterion_d, margin=opts.margin)

    # load generator
    generator = Generator(
        arch=arch_g, device=device,
        dim_latent=opts.dim_latent, 
        criterion=criterion_g,
        optimizer=optimizer_g,
        learning_policy=learning_policy_g
    )
    discriminator = Discriminator(
        arch=arch_d, device=device,
        criterion=criterion_d, 
        optimizer=optimizer_d,
        learning_policy=learning_policy_d
    )

    # generate the path for logging information and saving parameters
    cfg['info_path'], log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        generator=opts.generator,
        discriminator=opts.discriminator,
        description=opts.description
    )
    
    if opts.is_load_checkpoint:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path,
            models={"generator":generator, "discriminator":discriminator}
        )
    else:
        cfg['start_epoch'] = 0


    # load coach
    cfg['coach'] = Coach(
        generator=generator,
        discriminator=discriminator,
        device=device,
        normalizer=normalizer
    )

    return cfg, log_path



def main(
    coach, trainloader,
    start_epoch, info_path
):
    from freegan.utils import save_checkpoint, imagemeter
    for epoch in range(start_epoch, opts.epochs+1):
        if epoch % VALID_EPOCHS == 0:
            save_checkpoint(
                path=info_path,
                state_dict={
                    "generator":coach.generator.state_dict(),
                    "discriminator":coach.discriminator.state_dict(),
                    "epoch": epoch
                }
            )
            imgs = coach.generator.evaluate(batch_size=10)
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


    






    
