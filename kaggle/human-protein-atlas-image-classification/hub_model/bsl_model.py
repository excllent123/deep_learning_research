from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn

from collections import OrderedDict
import torch.nn.functional as F

def get_net(channels, num_classes):
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes),
            )
    return model
