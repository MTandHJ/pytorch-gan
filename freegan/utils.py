



import numpy as np
import torch
import os
import sys
from freeplot.utils import FreePlot



class AverageMeter:

    def __init__(self, name, fmt=".6f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1, mode="mean"):
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Sum: {sum:{fmt}} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, meters: AverageMeter, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch=8888):
        entries = [self.prefix + f"[Epoch: {epoch:<4d}]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def step(self):
        for meter in self.meters:
            meter.reset()

def imagemeter(*imgs):
    rows = len(imgs)
    imgs = [
        img.clone().detach().cpu().numpy().transpose((0, 2, 3, 1))
        for img in imgs
    ]
    cols = imgs[0].shape[0]
    fp = FreePlot((rows, cols), (cols, rows), dpi=100)
    for row in range(rows):
        for col in range(cols):
            fp.imageplot(imgs[row][col], index=row * cols + col)
    return fp.fig


def gpu(*models):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            model.to(device)
    return device

def mkdirs(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def readme(path, opts, mode="w"):
    """
    opts: the argparse
    """
    import time
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = path + "/README.md"
    s = "- {0[0]}:  {0[1]}\n"
    info = "\n## {0}".format(time_)
    for item in opts._get_kwargs():
        info += s.format(item)
    with open(filename, mode, encoding="utf8") as fh:
        fh.write(info)

# load model's parameters
def load(model, filename, device, strict=True, except_key=None):
    """
    :param model:
    :param filename:
    :param device:
    :param except_key: drop the correspoding key module
    :return:
    """

    if str(device) =="cpu":
        state_dict = torch.load(filename, map_location="cpu")
        
    else:
        state_dict = torch.load(filename)
    if except_key is not None:
        except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
        for key in except_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

# save the checkpoint
def save_checkpoint(path, state_dict):
    path = path + "/model-optim-lr_sch-epoch.tar"
    torch.save(
        state_dict,
        path
    )

# load the checkpoint
def load_checkpoint(path, models):
    path = path + "/model-optim-lr_sch-epoch.tar"
    checkpoints = torch.load(path)
    for key, model in models.items():
        checkpoint = checkpoints[key]
        model.load_state_dict(checkpoint)
    epoch = checkpoints['epoch'] + 1 # !
    return epoch

# caculate the lp distance along the dim you need,
# dim could be tuple or list containing multi dims.
def distance_lp(x, y, p, dim=None):
    return torch.norm(x-y, p, dim=dim)