import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 

import util
from hub_preprocess.dna_dataset import HumanDataset
from tqdm import tqdm 
import config as CONF

from datetime import datetime

from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start):
    losses = util.AverageMeter()
    f1 = util.AverageMeter()
    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        # compute output
        output = model(images)
        # for inceptionv3
        if type(output)==tuple:
            output=output[0]
        #output.view(-1), target.float()
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))
        
        f1_batch = f1_score(target,output.sigmoid().cpu() > 0.15,average='macro')
        f1.update(f1_batch,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg, 
                valid_loss[0], valid_loss[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                util.time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [losses.avg,f1.avg]

# 2. evaluate fuunction
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start):
    # only meter loss and f1 score
    losses = util.AverageMeter()
    f1 = util.AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            #image_var = Variable(images).cuda()
            #target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(images_var)
            loss = criterion(output,target)
            losses.update(loss.item(),images_var.size(0))
            f1_batch = f1_score(target,output.sigmoid().cpu().data.numpy() > 0.15,average='macro')
            f1.update(f1_batch,images_var.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    train_loss[0], train_loss[1], 
                    losses.avg, f1.avg,
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    util.time_to_str((timer() - start),'min'))

            print(message, end='',flush=True)
        log.write("\n")
        
    return [losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds, config):
    sample_submission_df = pd.read_csv("C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/sample_submission.csv")
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            #print(label > 0.5)
           
            labels.append(label > 0.15)
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('{}/{}_bestloss_submission.csv'.format(config.submit , config.model_name), index=None)

def oversample_multilabel(df):
    df['temp'] = [[int(i) for i in s.split()] for s in df['Target']]  
    multi = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
    #multi = [1,1,1,1,1,1,1,1,4,4,4,1,1,1,1,4,1,1,1,1,2,1,1,1,1,1,1,4]
    res_df = pd.DataFrame()
    for i in range(len(multi)):
        mask = df['temp'].apply(lambda x: i in x)
        temp_df = df[mask]
        pre = len(res_df)
        for j in range(multi[i]):
            res_df = res_df.append(temp_df, ignore_index=True)
        print('-> Oversample {} from {} samples to {} samples'.format(i, len(temp_df) ,len(res_df)-pre))
    del res_df['temp']
    res_df.index = range(len(res_df))
    return res_df

# 4. main function
def main(config):
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    
    # 4.2 get model
    model = config.model
    model.cuda()

    # criterion
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().cuda()
    #criterion = FocalLoss().cuda()
    #criterion = F1Loss().cuda()
    start_epoch = 0
    best_loss = 999
    best_f1 = 0
    best_results = [np.inf,0]
    val_metrics = [np.inf,0]

    train_df = pd.read_csv("C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/train.csv")
    #print(train_df)
    train_df = oversample_multilabel(train_df)
    test_files = pd.read_csv("C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/sample_submission.csv")
    
    # =============================== 
    # msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # train_df_orig = train_df.copy()
    # X = train_df_orig['Id'].tolist()
    # y = train_df_orig['target_vec_float'].tolist()
    # for train_index, test_index in msss.split(X,y): #it should only do one iteration
    #     train_data_list = 
    # ============================

    train_data_list,val_data_list = train_test_split(train_df,test_size = 0.1,
        random_state = 2050)

    # load dataset
    train_gen = HumanDataset(train_data_list,  config,mode="train")

    train_loader = DataLoader(train_gen,
        batch_size=config.batch_size,shuffle=True,
        pin_memory=True,num_workers=4)

    val_gen = HumanDataset(val_data_list, config,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,
        pin_memory=True,num_workers=4)

    test_gen = HumanDataset(test_files,config ,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    start = timer()

    #train
    for epoch in range(0,config.epochs):
        scheduler.step(epoch)
        # train
        lr = util.get_learning_rate(optimizer)
        train_metrics = train(train_loader,model,
            criterion,
            optimizer,
            epoch,
            val_metrics
            ,best_results,start)
        # val
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
        # check results 
        is_best_loss = val_metrics[0] < best_results[0]
        best_results[0] = min(val_metrics[0],best_results[0])
        is_best_f1 = val_metrics[1] > best_results[1]
        best_results[1] = max(val_metrics[1],best_results[1])   
        # save model
        util.save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_loss":best_results[0],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[1],
        },is_best_loss,is_best_f1,fold)
        # print logs
        print('\r',end='',flush=True)
        log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "best", epoch, epoch,                    
                train_metrics[0], train_metrics[1], 
                val_metrics[0], val_metrics[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                util.time_to_str((timer() - start),'min'))
            )
        log.write("\n")
        time.sleep(0.01)

    best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(
        config.best_models, config.model_name,str(fold)))
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold, config)

if __name__ == "__main__":
    # 1. set random seed
    random.seed(2050)
    np.random.seed(2050)
    torch.manual_seed(2050)
    torch.cuda.manual_seed_all(2050)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')    

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")    

    log = util.Logger()
    log.open("logs/%s_log_train.txt"%CONF.config.model_name,mode="a")
    log.write("\n [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
    log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')    
    # main
    main(CONF.config)
