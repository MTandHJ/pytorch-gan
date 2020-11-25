
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


from .dict2obj import Config
from .config import *



class ModelNotDefineError(Exception): pass
class LossNotDefineError(Exception): pass
class OptimNotIncludeError(Exception): pass
class DatasetNotIncludeError(Exception): pass


# return the num_classes of corresponding dataset
def get_num_classes(dataset_type: str):
    try:
        return NUMCLASSES[dataset_type]
    except KeyError:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


# return the shape of corresponding dataset
def get_shape(dataset_type: str):
    try:
        return SHAPES[dataset_type]
    except:
        raise DatasetNotIncludeError("Dataset {0} is not included." \
                        "Refer to the following: {1}".format(dataset_type, _dataset.__doc__))


def load_model(model_type: str):
    """
    gan-g: the generator defined in gan.py
    gan-d: the discriminator defined in gan.py
    dcgan-g: the generator defined in dcgan.py
    dcgan-d: the discriminator defined in the dcgan.py
    cgan-g: the generator defined in cgan.py
    cgan-d: the discriminator defined in the cgan.py
    """
    if model_type == "gan-g":
        from models.gan import Generator
        model = Generator
    elif model_type == "gan-d":
        from models.gan import Discriminator
        model = Discriminator
    elif model_type == "dcgan-g":
        from models.dcgan import Generator
        model = Generator
    elif model_type == "dcgan-d":
        from models.dcgan import Discriminator
        model = Discriminator
    elif model_type == "cgan-g":
        from models.cgan import Generator
        model = Generator
    elif model_type == "cgan-d":
        from models.cgan import Discriminator
        model = Discriminator
    elif model_type == "lapgan-g":
        from models.lapgan import Generator
        model = Generator
    elif model_type == "lapgan-d":
        from models.lapgan import Discriminator
        model = Discriminator
    elif model_type == "infogan-g":
        from models.infogan import Generator
        model = Generator
    elif model_type == "infogan-d":
        from models.infogan import Discriminator
        model = Discriminator
    else:
        raise ModelNotDefineError(f"model {model_type} is not defined.\n" \
                    f"Refer to the following: {load_model.__doc__}\n")
    return model


def load_loss_func(loss_type: str):
    """
    cross_entropy: the softmax cross entropy loss
    bce: binary cross entropy
    """
    if loss_type == "cross_entropy":
        from .loss_zoo import cross_entropy
        loss_func = cross_entropy
    elif loss_type == "bce":
        from .loss_zoo import bce_loss
        loss_func = bce_loss
    elif loss_type == "mse":
        from .loss_zoo import mse_loss
        loss_func = mse_loss
    else:
        raise LossNotDefineError(f"Loss {loss_type} is not defined.\n" \
                    f"Refer to the following: {load_loss_func.__doc__}")
    return loss_func


class _Normalize:

    def __init__(self, mean=None, std=None):
        self.set_normalizer(mean, std)

    def set_normalizer(self, mean, std):
        if mean is None or std is None:
            self.flag = False
            return 0
        self.flag = True
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.nat_normalize = T.Normalize(
            mean=mean, std=std
        )
        self.inv_normalize = T.Normalize(
            mean=-mean/std, std=1/std
        )

    def _normalize(self, imgs, inv):
        if not self.flag:
            return imgs
        if inv:
            normalizer = self.inv_normalize
        else:
            normalizer = self.nat_normalize
        new_imgs = [normalizer(img) for img in imgs]
        return torch.stack(new_imgs)

    def __call__(self, imgs, inv=False):
        # normalizer will set device automatically.
        return self._normalize(imgs, inv)


def _get_normalizer(dataset_type: str):
    mean = MEANS[dataset_type]
    std = STDS[dataset_type]
    return _Normalize(mean, std)


def _get_transform(dataset_type: str, transform: str, train=True):
    if train:
        return TRANSFORMS[dataset_type][transform]
    else:
        return T.ToTensor()


def _dataset(dataset_type: str, transform: str,  train=True):
    """
    Dataset:
    mnist: MNIST
    cifar10: CIFAR-10
    cifar100: CIFAR-100
    Transform:
    default: the default transform for each data set
    """
    try:
        transform = _get_transform(dataset_type, transform, train)
    except KeyError:
        raise DatasetNotIncludeError(f"Dataset {dataset_type} or transform {transform} is not included.\n" \
                        f"Refer to the following: {_dataset.__doc__}")

    if dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=ROOT, train=train, download=False,
            transform=transform
        )
    elif dataset_type == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=ROOT, train=train, download=False,
            transform=transform
        )
        
    return dataset


def load_normalizer(dataset_type: str):
    normalizer = _get_normalizer(dataset_type)
    return normalizer


def load_dataset(dataset_type: str, transform='default', train=True):
    dataset = _dataset(dataset_type, transform, train)
    return dataset


class _TQDMDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(_TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )


def load_dataloader(dataset, batch_size: int, train=True):
    if train:
        dataloader = _TQDMDataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)
    else:
        dataloader = _TQDMDataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY)

    return dataloader


def load_optimizer(
    model: torch.nn.Module, 
    optim_type: str, *,
    lr=0.1, momentum=0.9,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    nesterov=False,
    **kwargs
):
    """
    sgd: SGD
    adam: Adam
    """
    try:
        cfg = OPTIMS[optim_type]
    except KeyError:
        raise OptimNotIncludeError(f"Optim {optim_type} is not included.\n" \
                        f"Refer to the following: {load_optimizer.__doc__}")
    
    kwargs.update(lr=lr, momentum=momentum, betas=betas, 
                weight_decay=weight_decay, nesterov=nesterov)
    
    cfg.update(**kwargs) # update the kwargs needed automatically
    print(optim_type, cfg)
    if optim_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), **cfg)
    elif optim_type == "adam":
        optim = torch.optim.Adam(model.parameters(), **cfg)

    return optim


def load_learning_policy(
    optimizer: torch.optim.Optimizer,
    learning_policy_type: str,
    **kwargs
):
    """
    none: no learning policy
    6009: lr decays with a factor of 0.9 for each 60 epochs
    6002: lr decays with a factor of 0.2 for each 60 epochs
    resnet32-prime: the setting introduced in the prime paper for testing
                    on the CIFAR-10 under resnet20|32|44|56|110|1202
                    lr=0.1, optim=sgd
    cosine: CosineAnnealingLR, kwargs: T_max, eta_min, last_epoch
    """
    
        
    try:
        learning_policy_ = LEARNING_POLICY[learning_policy_type]
    except KeyError:
        raise NotImplementedError(f"Learning_policy {learning_policy_type} is not defined.\n" \
            f"Refer to the following: {load_learning_policy.__doc__}")

    lp_type = learning_policy_[0]
    lp_cfg = learning_policy_[1]
    lp_cfg.update(**kwargs) # update the kwargs needed automatically
    print(lp_type, lp_cfg)
    learning_policy = getattr(
        torch.optim.lr_scheduler, 
        lp_type
    )(optimizer, **lp_cfg)
    
    return learning_policy


def generate_path(
    method: str, dataset_type: str, 
    generator: str, discriminator: str,  
    description: str
):
    info_path = INFO_PATH.format(
        method=method,
        dataset=dataset_type,
        generator=generator,
        discriminator=discriminator,
        description=description
    )
    log_path = LOG_PATH.format(
        method=method,
        dataset=dataset_type,
        generator=generator,
        discriminator=discriminator,
        description=description
    )
    return info_path, log_path

