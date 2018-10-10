from __future__ import print_function

from glob import glob
import pandas as pd 
import torch
from skimage.transform import rescale, resize, rotate, AffineTransform
import numpy as np 
import os 
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread, imshow
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
import sys 
from tqdm import tqdm


class DefRefDataset_DF(Dataset):
    '''
    2018 t-kaggle 

    Args:
      - dir_pth : with label as sub-folder

    '''
    def __init__(self, hook_table, trans_ops=[], 
        img_width=400, img_height=400):
        
        super(DefRefDataset_DF, self).__init__()

        self.hook_table = hook_table

        key = ['0', '118', '134', '142', '145', '146', '148', '158', '159', '163', '164']
        value = range(len(key))
        one_hot = np.eye(len(key))
        map_dict = {}
        for k, v in list(zip(key, value)):
            map_dict[k] = one_hot[v]
        
        self.map_dict = map_dict
        self.trans_ops = trans_ops

        self.img_width = img_width
        self.img_height = img_height
        
    def __len__(self):
        return len(self.hook_table)
    
    def _img_nor(self, img):
        img = img.astype('float32')/255. 
        img = img - 0.5       
        return img 

    def __getitem__(self, idx):
        row = self.hook_table.iloc[idx]

        def_img = imread(row['def_files'])
        ref_img = imread(row['ref_files'])

        def_img, ref_img = image_alignment(def_img, ref_img)

        label   = row['class']
        label   = self.map_dict[label]
        if self.trans_ops:
            for trans_op in self.trans_ops:
                def_img, ref_img = trans_op(def_img, ref_img)
 
        def_img = resize(def_img,(self.img_width, self.img_height))
        ref_img = resize(ref_img,(self.img_width, self.img_height))
        dff_img = extract_diff(def_img, ref_img)  
        


        #dff_img[:, :, 0] = np.min(dff_img[:, :, 0] + dff_img[:, :, 1], 1)
        dff_img[:, :, 1] = ref_img[:, :, 2]

        def_img = np.transpose(def_img, (2, 1, 0))
        ref_img = np.transpose(ref_img, (2, 1, 0)) 
        dff_img = np.transpose(dff_img, (2, 1, 0))   

        return dff_img, label, row['def_files']


def get_train_df(dir_pth  = 'd:/_08_test_data/TSMC__kaggle_2018/training_labeled//'):
    
    files = [i for i in glob(dir_pth+'\**', recursive=True ) ]
            
    ref_files = [i for i in files if i[-7:] =='ref.jpg']
    def_files = [i for i in files if i[-7:] =='def.jpg']
                

    labels_ref   = [ i.split('\\')[-2] for i in ref_files]
    labels_def   = [ i.split('\\')[-2] for i in def_files]    

    res_df = pd.DataFrame()
    res_df["class"] = labels_ref
    res_df["ref_files"] = ref_files
    res_df["def_files"] = def_files
    return res_df

def hook_df_balance_target(df, tar_col):
    count_df = df[tar_col].value_counts()
    count_mean = count_df.mean()
    for value in  list(set(df[tar_col])):
        temp_df = df[df[tar_col]==value]
        while len(df[df[tar_col]==value]) < count_mean:
            df = df.append(temp_df, ignore_index=True)
    return df

def gen_oof_dataset(hook_table, target_col,  n_fold, trans_ops, 
    starfied=True, img_width=400, img_height=400):
    '''
    could custonmized trian-test by Kfold or ? 
    trans_ops= augmentation - only in train
    '''
    if starfied: 
        fold = StratifiedKFold(n_fold)
        iter_ = fold.split(hook_table, hook_table[target_col])
        print('starfied')
    else: 
        fold = KFold(n_fold)
        iter_ = fold.split(hook_table)
        print('not starfied')

    for train_ind, valid_ind in iter_:
        hook_table['train_valid'] = 'train'
        hook_table['train_valid'].iloc[valid_ind] = 'valid'
        train_dataset = DefRefDataset_DF(hook_table[hook_table['train_valid']=='train'], 
            trans_ops=trans_ops, 
            img_width=img_width, 
            img_height=img_height)
        vaild_dataset = DefRefDataset_DF(hook_table[hook_table['train_valid']=='valid'], 
            trans_ops=[], # valid dataset do not augment 
            img_width=img_width, 
            img_height=img_height)
        yield train_dataset, vaild_dataset, hook_table

def train_single_fold(train_dataset, 
    vaild_dataset, model, criterion, optimizer, 
    model_export_dir, batch_size=16,
    test_mode=False):
    '''
    model export dir 
    '''
    # ====================================
    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor
    # ====================================
    model.cuda()    

    epoch = 1
    epoch_max_tolerance = 5
    epoch_tolerance = 0
    res_f_score = 0
    while epoch_tolerance < epoch_max_tolerance:
        update_time = str(datetime.today()).split('.')[0].replace(':', '-')
        record_df = pd.DataFrame()
        predict_df = pd.DataFrame()    

        for train_mode in ['train', 'valid']:
            running_loss = 0.0

            if train_mode =='train':
                model.train()    

                iter_item = tqdm(DataLoader(train_dataset, 
                                batch_size=batch_size, num_workers=32 ), ascii=True)

                for diff_img, label, def_path in iter_item:
                    optimizer.zero_grad()
                    outputs =model(diff_img.type(Tensor))    

                    if test_mode:
                        print("=====================")
                        print(outputs, label)

                    # Input: (N,C) where C = number of classes
                    # Target: (N) where each value is 0 <= targets[i] <= C-1
                    loss = criterion (outputs.type(Tensor), label.type(Tensor))
                    running_loss += loss.item()

                    # backward + optimize  
                    loss.backward()
                    optimizer.step()
                    if test_mode: break

                    iter_item.set_description('Loss %.4f' %(loss.item()))
                
                print('epoch : %d, loss: %.3f' %(epoch,
                                                 running_loss / len(train_dataset)))

            else :
                model.eval()
                running_loss = 0.0
                all_f_score = 0.0    

                iter_item = tqdm(DataLoader(vaild_dataset, 
                    batch_size=batch_size, num_workers=32 ), ascii=True)
                with torch.no_grad():
                    for diff_img, label, def_path in iter_item:
                        optimizer.zero_grad()
                        outputs =model(diff_img.type(Tensor))     

                        loss = criterion(outputs.type(Tensor), label.type(Tensor))

                        running_loss += loss.item()

                        f_score = fbeta_score(outputs.cpu(), label.cpu() )        
                        all_f_score   += f_score
                        if test_mode: break
                        iter_item.set_description('valid-loss {}'.format(loss.item()))

                val_f_score = round(all_f_score.numpy()/len(vaild_dataset), 4)

                print('epoch {}, valid-loss : {},  valid_f1 : {}'.format( 
                       epoch, 
                       running_loss / len(vaild_dataset),
                       val_f_score))    

        
                file_name = str(val_f_score) + "_" + str(epoch) + '_checkpoint.pth'
                is_best = None
                save_checkpoint({
                          'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                        }, is_best,  file_name, 
                        save_path=model_export_dir)        

                record_df = record_df.append({            

                    'epoch':epoch,
                    'model_pth': '{}/'.format(model_export_dir)+ file_name,
                    'f_score': val_f_score,
                    'update_time' : update_time,
                    }, ignore_index=True)        

                record_df.to_csv('record_df.csv', mode='a', index=False, header=False)
                print('save record ')
                if val_f_score < res_f_score :
                    epoch_tolerance+=1 
                else: 
                    epoch_tolerance = 0 
                res_f_score = max(res_f_score, val_f_score)
                epoch+=1

def fbeta_score(y_true, y_pred, beta=1, threshold=0.5, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

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

def image_alignment(imgDef0, imgRef0, img_wid=299, img_hei=299):
    imgDef = imgDef0[:,:,0]*0.299 + imgDef0[:,:,1]*0.587 + imgDef0[:,:,2]*0.114
    imgRef = imgRef0[:,:,0]*0.299 + imgRef0[:,:,1]*0.587 + imgRef0[:,:,2]*0.114
    
    ### coarse
    imgDefs = resize(imgDef,(50, 50))
    imgRefs = resize(imgRef,(50, 50))
    difference = np.zeros((11,11))
    landmark = imgDefs[5:45,5:45]
    # imshow(landmark)
    
    for i in range(11):
        for j in range(11):
            difference[i,j] = np.sum(np.abs(landmark - imgRefs[i:i+40,j:j+40]))
    
    index = np.argmin(difference)
    ny = index//11
    nx = index%11
    dy0 = -(ny - 5)*6
    dx0 = -(nx - 5)*6
    
    ### fine
    difference = np.zeros((11,11))
    landmark = imgDef[45+dy0:244+dy0,45+dx0:244+dx0]
    
    for i in range(11):
        for j in range(11):
            difference[i,j] = np.sum(np.abs(landmark - imgRef[i+40:i+239,j+40:j+239]))
    
    index = np.argmin(difference)
    ny = index//11
    nx = index%11
    dy = -(ny - 5) + dy0
    dx = -(nx - 5) + dx0
      
    y1Def = max(0,dy)
    y2Def = min(img_hei,img_hei+dy)
    x1Def = max(0,dx)
    x2Def = min(img_wid,img_wid+dx)

    y1Ref = max(0,-dy)
    y2Ref = min(img_hei,img_hei-dy)
    x1Ref = max(0,-dx)
    x2Ref = min(img_wid,img_wid-dx)

    imgDef = resize(imgDef0[y1Def:y2Def,x1Def:x2Def],(img_hei,img_wid))
    imgRef = resize(imgRef0[y1Ref:y2Ref,x1Ref:x2Ref],(img_hei,img_wid))
    
    return imgDef, imgRef

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

def extract_diff(def_img, ref_img):
    diff = def_img - ref_img
    diff = np.vectorize(lambda x : abs(x))(diff)
    thresh = np.median(diff)
    diff = np.vectorize(lambda x : 0. if x < thresh else x)(diff)
    return diff

def difference_proprose(def_img, ref_img):
    '''
    should be easy find difference region by sliding window 
    '''
    pass



# def image_augmentation(imgDef, imgRef, imgDif, img_wid, img_hei):
#     rand = np.random.randint(0,2)
#     if rand == 1:
#         M90 = cv2.getRotationMatrix2D((149, 149), 90, 1.0)
#         imgDef = cv2.warpAffine(imgDef, M90, (img_wid, img_hei))
#         imgRef = cv2.warpAffine(imgRef, M90, (img_wid, img_hei))
#         imgDif = cv2.warpAffine(imgDif, M90, (img_wid, img_hei))

#     rand = np.random.randint(0,2)
#     if rand == 1:
#         imgDef = cv2.flip(imgDef, 0)
#         imgRef = cv2.flip(imgRef, 0) 
#         imgDif = cv2.flip(imgDif, 0) 

#     rand = np.random.randint(0,2)
#     if rand == 1:
#         imgDef = cv2.flip(imgDef, 1)
#         imgRef = cv2.flip(imgRef, 1) 
#         imgDif = cv2.flip(imgDif, 1)
#     
#     rand = np.random.randint(75,300)
#     if rand <= 149:
#         imgDef = imgDef[149-rand:149+rand,149-rand:149+rand,:]
#         imgDef = cv2.resize(imgDef, (img_wid, img_hei)) 
#         imgRef = imgRef[149-rand:149+rand,149-rand:149+rand,:]
#         imgRef = cv2.resize(imgRef, (img_wid, img_hei))
#         imgDif = imgDif[149-rand:149+rand,149-rand:149+rand,:]
#         imgDif = cv2.resize(imgDif, (img_wid, img_hei))
#     else:
#         imgDef0 = np.zeros((1+2*rand,1+2*rand,3))
#         imgDef0[rand-149:rand+150,rand-149:rand+150,:] = imgDef
#         imgDef = cv2.resize(imgDef0, (img_wid, img_hei))
#         imgRef0 = np.zeros((1+2*rand,1+2*rand,3))
#         imgRef0[rand-149:rand+150,rand-149:rand+150,:] = imgRef
#         imgRef = cv2.resize(imgRef0, (img_wid, img_hei))
#         imgDif0 = np.zeros((1+2*rand,1+2*rand,3))
#         imgDif0[rand-149:rand+150,rand-149:rand+150,:] = imgDif
#         imgDif = cv2.resize(imgDif0, (img_wid, img_hei))
#     return imgDef, imgRef, imgDif
