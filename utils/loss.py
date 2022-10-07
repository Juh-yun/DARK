import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class CrossEntropy_SL_LS(object):
    def __init__(self, num_class=126, ls=0.1):
        self.num_class = num_class
        self.ls = ls

    def __call__(self, pred, pseudo_label, weight):
        logsoftmax = F.log_softmax(pred, dim=1)
        ce = ((((pseudo_label + self.ls * (1.0/self.num_class)) / (1.0+self.ls)) * -logsoftmax).sum(dim=1) * weight).mean()
        return ce


class CrossEntropy_SL(object):
    def __init__(self, num_class=126):
        self.num_class = num_class
    def __call__(self, pred, label, weight):
        logp = F.log_softmax(pred, dim=1)
        ce = ((label * -logp).sum(dim=1) * weight).mean()
        return ce


class CrossEntropy_LS(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropy_LS, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class CrossEntropy(object):
    def __init__(self, num_class=126):
        self.num_class = num_class

    def __call__(self, pred, pseudo_label, mask):
        ce = (F.cross_entropy(pred, pseudo_label, reduction='none') * mask).mean()
        return ce