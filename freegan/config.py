


# Here are some basic settings.
# It could be overwritten if you want to specify
# special configs. However, please check the correspoding
# codes in loadopts.py.



import torchvision.transforms as T
from .dict2obj import Config




ROOT = "../data"
INFO_PATH = "./infos/{method}/{dataset}-{generator}-{discriminator}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{generator}-{discriminator}/{description}"
TIMEFMT = "%m%d%H"


# basic properties of inputs
MEANS = {
    "mnist": None,
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408],
    "svhn": [0.5071, 0.4867, 0.4409]
}

STDS = {
    "mnist": None,
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761],
    "svhn": [0.2675, 0.2565, 0.2761]
}


TRANSFORMS = {
    "mnist": {
        'default': T.ToTensor()
    },
    "cifar10": {
        'default': T.ToTensor()
    }
}
TRANSFORMS["cifar100"] = TRANSFORMS["cifar10"]



# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=False),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0.)
}


# the learning schedular can be added here
LEARNING_POLICY = {
    "none":(
        "StepLR",
        Config(
            step_size=999999,
            gamma=1.
        )
    ),
    "6009":(
        "StepLR",
        Config(
            step_size=60,
            gamma=0.9
        )
    ),
    "6002":(
        "StepLR",
        Config(
            step_size=60,
            gamma=0.2
        )
    ),
    "resnet32-prime":(   # the schedular introduced in the prime paper
        "MultiStepLR",   # check that optimizer sgd is matched.
        Config(
            milestones=[82, 123],
            gamma=0.1
        )
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
        )
    )
}




