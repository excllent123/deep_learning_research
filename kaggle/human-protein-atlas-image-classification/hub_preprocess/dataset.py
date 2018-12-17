
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import KFold, StratifiedKFold
from skimage.transform import rotate, resize, rescale
import time

import copy
from tqdm import tqdm

from utils import *
from config import *
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import random
import pathlib

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class HumanDataset(Dataset):
    def __init__(self,images_df,base_path,augument=True,mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.images_df = images_df.copy()
        self.augument = augument
        self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / x)
        self.mlb = MultiLabelBinarizer(classes = np.arange(0,config.num_classes))
        self.mlb.fit(np.arange(0,config.num_classes))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        #X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])(X)
        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X)
        return X.float(),y


    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        #use only rgb channels
        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else:
            images = np.zeros(shape=(512,512,3))
        r = np.array(Image.open(filename+"_red.png")) 
        g = np.array(Image.open(filename+"_green.png")) 
        b = np.array(Image.open(filename+"_blue.png")) 
        y = np.array(Image.open(filename+"_yellow.png")) 
        images[:,:,0] = r.astype(np.uint8) 
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = b.astype(np.uint8)
        if config.channels == 4:
            images[:,:,3] = y.astype(np.uint8)
        images = images.astype(np.uint8)
        #images = np.stack(images,-1) 
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))

    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

class RSNA_Dataset(Dataset):
    '''
    Return img, x, y, width, height, PatientSex, PatientAge, ViewPosition, Target
    '''
    def __init__(self, hook_df):

        self.img_augment=img_augment
        # if you want inference => not really need a batch-pumping 
        self.hook_df = hook_df[hook_df['train_valid']==train_valid].reset_index()


    def __getitem__(self, idx): 
        row = self.hook_df.iloc[idx]
        img = pydicom.dcmread(row['pth']).pixel_array
        img = img.astype('float32')
        x             = row['x']
        y             = row['y']
        width         = row['width']
        height        = row['height']
        PatientSex    = row['PatientSex']
        PatientAge    = row['PatientAge']
        ViewPosition  = row['ViewPosition']
        Target        = row['Target'].astype('float32')

        img /=255.
        img -=0.5
        if self.img_augment:
            img = self.img_augment(img)    
        if self.img_augment_box:
            img, x, y, width, height = self.img_augment_box(img, x, y, width, height)    
        img = resize(img, (self.trian_img_h, self.train_img_w))
        img = np.reshape(img, (1, self.trian_img_h, self.train_img_w))

        return img , x, y, width, height, PatientSex, PatientAge, ViewPosition, Target

    def __len__(self):
        return len(self.hook_df)

def image_augment(def_img, ref_img):
    raw_h, raw_w = def_img.shape[0], def_img.shape[1]
    # orient-augmentation 
    rand = np.random.randint(0,10)
    if rand > 4:
        def_img = rotate(def_img, 90)
        ref_img = rotate(ref_img, 90)

    # small angle-jitter
    rand = np.random.randint(0,10)
    if rand < 5:
        def_img = rotate(def_img, rand)
        ref_img = rotate(ref_img, rand)

    rand = np.random.randint(0,10)
    if rand > 4:
        def_img = def_img[ :, ::-1, :  ]
        ref_img = ref_img[ :, ::-1, :  ]

    rand = np.random.randint(0,10)
    if rand > 4:
        def_img = def_img[ ::-1, : , : ]
        ref_img = ref_img[ ::-1, : , : ]

    rand = np.random.randint(0, 60)
    if rand < 40:
        def_img = def_img[rand :raw_h-rand, rand : raw_w-rand,:]
        ref_img = ref_img[rand :raw_h-rand, rand : raw_w-rand,:]

        def_img = resize(def_img,(raw_h,raw_w) )
        ref_img = resize(ref_img,(raw_h,raw_w) )
    return def_img, ref_img

    
def img_augment(img):
    '''
    2d
    '''
    rand = np.random.randint(0, 180)
    if rand < 90:
        img = rotate(img, rand)

    rand = np.random.randint(0, 2)
    if rand < 1:
        img = img[::-1, :]

    rand = np.random.randint(0, 2)
    if rand < 1:
        img = img[:, ::-1 ]

    rand = np.random.randint(0, 200)
    if rand < 100:
        h, w = img.shape
        img = img[rand:h-rand, rand:w-rand]
    return img

def get_metadata_per_patient(file_pth, attribute):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata =  pydicom.read_file(file_pth)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value


def get_hook_df():
    train_label = pd.read_csv('C:/Users/kent/.kaggle/competitions/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
    trian_img_dir = 'C:/Users/kent/.kaggle/competitions/rsna-pneumonia-detection-challenge/stage_1_train_images/'    

    train_label['pth'] = train_label.patientId.apply(lambda x : trian_img_dir+x+'.dcm')     

    attributes = ['PatientSex', 'PatientAge', 'ViewPosition' ]
    for i in attributes:
        train_label[i]  = train_label['pth'].apply(lambda x: get_metadata_per_patient(x, i))

    train_label = train_label.fillna(0)
    return train_label

def gen_torch_dataset(n_fold, train_img_h, train_img_w, img_augment, img_augment_box):
    '''
    could custonmized trian-test by Kfold or ? 
    '''
    try:
        hook_table = pd.read_feather('hook_df2.feather')
    except:
        hook_table = get_hook_df()
        hook_table.to_feather('hook_df2.feather')

    fold = KFold(n_fold)
    for train_ind, valid_ind in fold.split(hook_table):
        hook_table['train_valid'] = 'train'
        hook_table['train_valid'].iloc[valid_ind] = 'valid'
        train_dataset = RSNA_Dataset(hook_table, 
                                     train_img_h, train_img_w, 
                                     img_augment=img_augment, 
                                     img_augment_box=img_augment_box, 
                                     train_valid ='train')

        vaild_dataset = RSNA_Dataset(hook_table, 
                                     train_img_h, train_img_w, 
                                     img_augment=[], 
                                     img_augment_box=img_augment_box,
                                     train_valid ='valid')
        yield train_dataset, vaild_dataset, hook_table

def save_checkpoint(state, is_best, filename, save_path='model_export'):
    '''
    Usage : 
    ```python 
        save_checkpoint({
                  'epoch': epoch,
                  # 'arch': args.arch,
                  'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict(),
                }, is_best, mPath ,  str(val_acc) + '_' + \
                str(val_los) + "_" + str(epoch) + '_checkpoint.pth.tar')
    ```
    '''
    if not os.path.exists(save_path):  os.makedirs(save_path)

    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def load_checkpoint(load_path, model, optimizer):
    """ loads state into model and optimizer and returns:
        epoch, best_precision, loss_train[]
    """
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch']

        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(epoch, checkpoint['epoch']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_path))

def train_whether_abnormal(model, gen_dataset, criterion, optimizer, scheduler, batch_size, 
    n_fold ,model_export_dir,  num_epochs=25, test_mode=False):
    '''
    Args :
        gen_dataset return train_dataset, vaild_dataset, hook_table
    '''
    since = time.time() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    res_oof_model = []
    res_oof_loss = []
    res_oof_df = []

    fold_numb = 1

    for train_dataset, vaild_dataset, hook_table in gen_dataset:
        print('Fold ', fold_numb)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)    

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                    dataset = train_dataset
                else:
                    model.eval()   # Set model to evaluate mode    
                    dataset = vaild_dataset

                running_loss = 0.0
                running_corrects = 0    

                # Iterate over data.
                for packages in tqdm(DataLoader(dataset, batch_size=batch_size)):
                    img, x, y, width, height, PatientSex, PatientAge, ViewPosition, Target = packages
                    img = img.float().to(device)
                    Target = Target.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()    

                    # forward
                    # track history if only in train
                    #with torch.set_grad_enabled(phase == 'train'):
                    

                    outputs = model(img)
                    time_elapsed = time.time() - since

                    loss = criterion(outputs, Target.view(-1, 1),)    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()    



                    # statistics
                    running_loss += loss.item() * img.size(0)
                    running_corrects += torch.sum(preds.float() == Target.data)   
                    if test_mode : break 

                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects.double() / len(dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))    

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())    
                    model_save_name = 'fold-{}_epoch-{}_acc-{}.pkl'.format(fold_numb, epoch, best_acc)
                    # model.save_state_dict(model_save_name)

                    save_checkpoint({
                              'epoch': epoch,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                            }, is_best=None,  filename=model_save_name, 
                            save_path=model_export_dir)  
 


            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))    

        # load best model weights
        model.load_state_dict(best_model_wts)

        res_oof_model.append(model)
        res_oof_loss.append(best_acc)
        res_oof_df.append(hook_table)
        fold_numb +=1

    return res_oof_model, res_oof_loss, res_oof_df


