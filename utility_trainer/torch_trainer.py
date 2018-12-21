from __future__ import print_function

import os, sys
import numpy as np

# sys.path.append('../../')
from scipy import misc
import pandas as pd

from core_kernel.torch_modules import (util_visual, 
	                                   resnet, norm_resnet, util_preprocess, para_config)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split, SubsetRandomSampler
from torch import optim
from skimage.transform import AffineTransform, rescale, resize
from skimage.io import imread, imshow, imsave
from datetime import datetime
import score
from tqdm import tqdm 
import gc 
import warnings
warnings.filterwarnings("ignore")


def validate_epch(dataset, model, batch_size=64 ):
    '''
    Arg : 
      - dataset : customized pytorch dataset 
      - model : torch model 
    '''
    def one_hot_to_label(x, map_dict):
        '''
        map_dict = [key as np.argmax, value ='label']
        '''
        x = x.cpu().detach().numpy()
        x = list(np.argmax(x, 1))
        x = [map_dict[i] for i in x]    
        return x

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    map_dict ={}
    for k , v in dataset.map_dict.items(): map_dict[ np.argmax(v) ] = k    

    res_dict = {'def_img_pth' : [], 'predict': [], 'true': [], 'outputs':[]}

    model.eval()
    with torch.no_grad():
        for diff_img, label, def_path in tqdm(DataLoader(
            dataset, batch_size=32, num_workers=25),  ascii=True):

            outputs =model(diff_img.type(Tensor))            

            pre_lable = one_hot_to_label(outputs, map_dict)
            label = one_hot_to_label(label, map_dict)        

            def_path = list(def_path)

            res_dict['def_img_pth'] +=  def_path
            res_dict['predict'    ] +=  pre_lable
            res_dict['true'       ] +=  label
            res_dict['outputs'] += list(outputs.cpu().detach().numpy())

    return res_dict, map_dict

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def train_single_fold(train_dataset, 
    vaild_dataset, model, optimizer,
    model_export_dir,
    criterion, 
    aug_loss=False, 
    contrastive_loss=False, 
    batch_size=16,
    test_mode=False):
    '''
    model export dir 
    '''
    # ====================================
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    # ====================================
    model.cuda()    

    epoch = 1
    epoch_max_tolerance = 35
    epoch_tolerance = 0
    res_f_score = 0

    loss_regist = {'BCE': nn.BCEWithLogitsLoss(), 'CrossEntropyLoss' : nn.CrossEntropyLoss()}

    if criterion not in loss_regist.keys():
        raise ValueError('criterion must be BCE or CrossEntropyLoss')

    while epoch_tolerance < epoch_max_tolerance:
        update_time = str(datetime.today()).split('.')[0].replace(':', '-')
        record_df = pd.DataFrame()
        predict_df = pd.DataFrame()    

        for train_mode in ['train', 'valid']:
            running_loss = 0.0

            if train_mode =='train':
                model.train()    

                iter_item = tqdm(DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                num_workers=25, 
                                shuffle=True ), ascii=True)

                for diff_img, label, def_path in iter_item:
                    optimizer.zero_grad()
                    outputs =model(diff_img.type(Tensor))    

                    if aug_loss:
                        rand = np.random.randint(0,10)
                        if rand > 7.5:
                            criterion ='BCE'
                        else:
                            criterion = 'CrossEntropyLoss'

                    if criterion =='BCE':
                        loss = loss_regist[criterion](outputs.type(Tensor), label.type(Tensor))
                    elif criterion =='CrossEntropyLoss':
                        label = np.argmax(label, 1)
                        loss = loss_regist[criterion](outputs.type(Tensor), label.type(LongTensor))   
                    
                    running_loss += loss.item()

                    # backward + optimize  
                    loss.backward()
                    optimizer.step()
                    if test_mode: break

                    iter_item.set_description('Loss %.4f' %(loss.item()))
                
                print('epoch : %d, loss: %.4f' %(epoch,
                                                 running_loss / len(train_dataset)))

            else :
                model.eval()

                res_dict, map_dict = validate_epch(vaild_dataset, model)    
                matric, labels, contribution_score, purity_score, final_score = \
                    score.port_f_score(res_dict['predict'], 
                                       res_dict['true'], 
                                       list(map_dict.values()))
    
                print(contribution_score, purity_score, final_score)        
                file_name = str(final_score) + "_" +\
                            str(purity_score) + "_" +\
                            str(contribution_score) + "_" +\
                            str(epoch) + '.pth'
                is_best = None
                util_preprocess.save_checkpoint({
                          'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                        }, is_best,  file_name, 
                        save_path=model_export_dir) 

                # ==============================================
                # save check point validation for blended oof 
                valid_df_cpts = pd.DataFrame(res_dict)
                valid_df_cpts.to_csv(model_export_dir+'/'+
                                    file_name.replace('.pth', '.csv'), 
                                    index=False) 

                # ==============================================
                if final_score < res_f_score :
                    epoch_tolerance+=1 
                else: 
                    epoch_tolerance = 0 
                res_f_score = max(res_f_score, final_score)
                epoch+=1

def train_oof(CONF_FLAG, model_check_pt=None):

    # due to model capture at the class, need reinit

    hook_table = util_preprocess.get_train_df(CONF_FLAG().data_dir)
    hook_table = util_preprocess.hook_df_balance_target(hook_table, 'class',)

    # target_col,  n_fold, trans_ops, dataset_class
    oof_generator = util_preprocess.gen_oof_dataset(  hook_table     = hook_table, 
                                  target_col     = CONF_FLAG().target_col, 
                                  n_fold         = CONF_FLAG().n_fold, 
                                  dataset_class  = CONF_FLAG().dataset_class,
                                  trans_ops      = [util_preprocess.image_augment], 
                                  starfied       = CONF_FLAG().starfied, 
                                  img_height     = CONF_FLAG().train_img_h, 
                                  img_width      = CONF_FLAG().train_img_w )
    oof  = 1
    for train_dataset, vaild_dataset, hook_table in oof_generator:
        model_export_dir= '{}_fold-{}'.format(CONF_FLAG().model_export_dir, oof)
        try : 
            del model 
            gc.collect()
        except:
            pass 
        model = CONF_FLAG().model # meed reinit
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = 0.007)

        if model_check_pt:
            util_preprocess.load_checkpoint(model_check_pt, model, optimizer)
            print('loaded model')
        
        train_single_fold(train_dataset, 
            vaild_dataset, 
            model, 
            criterion = CONF_FLAG().criterion, 
            optimizer=optimizer, 
            aug_loss = CONF_FLAG().aug_loss,
            model_export_dir=model_export_dir, 
            batch_size=CONF_FLAG().batch_size, 
            test_mode=CONF_FLAG().test_mode)
        oof +=1


if __name__ =='__main__':
    train_oof(para_config.Hypara_16, 
        'models_cpts/Hypara_13_fold-1/1.9635_0.9676_0.9959_217.pth' 
         )
