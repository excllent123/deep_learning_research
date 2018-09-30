import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize
from skimage.io import imshow
from util_preprocess import get_hook_df, gen_torch_dataset, img_augment, train_whether_abnormal
from se_module import se_resnet18, SEBasicBlock
from model_hub import resnet18
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn 
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 10
n_fold = 4
num_epochs = 25
gen_dataset = gen_torch_dataset(n_fold=4, 
                                train_img_h=1024, 
                                train_img_w=1024, 
                                img_augment=img_augment, 
                                img_augment_box=None)

model = resnet18( num_classes=1, in_channels=1)
model = model.to(device)

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
                       num_epochs = num_epochs
)