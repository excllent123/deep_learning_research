from torchvision import models
from pretrainedmodels.models import bninception
from pretrainedmodels.models import inceptionv3, resnet34
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

def get_inceptionv3(channels, num_classes):
    model = inceptionv3(pretrained="imagenet")
    #model.global_pool = nn.AdaptiveAvgPool2d(1)
    # change layer... 
    model.Conv2d_1a_3x3 = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(2, 2), )
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )
    return model

def get_resnet34(channels, num_classes):
    model = resnet34(pretrained="imagenet")
    #model.global_pool = nn.AdaptiveAvgPool2d(1)
    # change layer... 
    model.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3) )
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
    return model

if __name__=='__main__':
    model = resnet34(pretrained="imagenet")
    #model.global_pool = nn.AdaptiveAvgPool2d(1)
    #model.con2d_1a_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    print(model)