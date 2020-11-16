

import torch
import torch.nn as nn
import torch.nn.functional as F



def cross_entropy(logits, targets, reduction="mean"):
    return F.cross_entropy(logits, targets, reduction=reduction)

def bce_loss(probs, targets, reduction="mean"):
    return F.binary_cross_entropy(probs, targets, reduction=reduction)

def mse_loss(x, y, reduction="mean"):
    return F.mse_loss(x, y, reduction=reduction)