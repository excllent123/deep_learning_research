import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from skimage.io import imshow
from util_preprocess import  gen_torch_dataset, img_augment, train_whether_abnormal
from se_module import se_resnet18, SEBasicBlock
from model_hub import resnet18, resnet14
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn 
from torch import optim
import os 
from summary import summary


batch_size = 64
n_fold = 5
num_epochs = 25
test_mode = False
train_img_h = 512
train_img_w = 512 
channel = 1

gen_dataset = gen_torch_dataset(n_fold=4, 
                                train_img_h=train_img_h, 
                                train_img_w=train_img_w, 
                                img_augment=img_augment, 
                                img_augment_box=None)

model = resnet14( num_classes=1, in_channels=1, fc_factor=800, base_ly_number=4)
model = model.cuda()

summary(model, (channel, train_img_h, train_img_w))

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

train_whether_abnormal(model, 
                       gen_dataset = gen_dataset, 
                       criterion = criterion,
                       optimizer  = optimizer_ft, 
                       scheduler  = exp_lr_scheduler,
                       batch_size = batch_size,
                       n_fold     = n_fold, 
                       num_epochs = num_epochs, 
                       model_export_dir = 'resnet18_c1', 
                       test_mode=test_mode, 
)