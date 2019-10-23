#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:53:27 2019

@author: dhanley2
"""
import numpy as np
import math
import pandas as pd
import os
from sklearn.metrics import log_loss
import ast
import pickle

from torchvision import transforms as T
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize
                           )

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

path='/Users/dhanley2/Documents/Personal/rsna/eda'

trnmdf = pd.read_csv(os.path.join(path, '../data/train_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1

wts3 = np.array([0.6, 1.8, 0.6])
#wts5 = np.array([0.5, 1., 2., 1., 0.5])
def f(w3):                        
    def g(x):
        if len(x)>2:
            return (w3*x).mean()
        else:
            return x.mean()
    return g

 

# Ct 0 LLoss 0.07719 RmeanLLoss 0.07233 LLoss Avg 0.07719 Rmean Avg 0.07233
ypredls = []
ypredrmeanls = []
for fold in [0]:
    yact = pd.read_csv(os.path.join(path, 'val_act_fold{}.csv.gz'.format(fold )))
    yactf = yact[label_cols].values.flatten()
    for epoch in range(0, 7):
        #ypred = pd.read_csv(os.path.join(path, 'seq/se50v3/val_pred_sz448_wt448_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        #ypred = pd.read_csv(os.path.join(path, '../../eda/seq/se100v1/val_pred_sz384_wt384_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        ypred = pd.read_csv(os.path.join(path, 'seq/v6/val_pred_sz384_wt384_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
        ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
        ypred = ypred.sort_values(['PatientID', 'seq'])
        ypredrmean = ypred.groupby('PatientID')[label_cols]\
                        .rolling(3, center = True, min_periods=1).apply(f(wts3)).values
        ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
        ypred = ypred.sort_index()
        ypredrmean = ypredrmean.sort_index()
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}_samp1.0.csv.gz'.format(i)))
        weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
        ypred = ypred[label_cols].values.flatten()
        ypredrmean = ypredrmean[label_cols].values.flatten()
        ypredls.append(ypred)    
        ypredrmeanls.append(ypredrmean)
        print('Epoch {} LLoss {:.5f} RmeanLLoss {:.5f} LLoss Avg {:.5f} Rmean Avg {:.5f}'.format(1+epoch, \
            log_loss(yactf, ypred, sample_weight = weights), \
            log_loss(yactf, ypredrmean, sample_weight = weights), \
            log_loss(yactf, sum(ypredls)/len(ypredls), sample_weight = weights), \
            log_loss(yactf, sum(ypredrmeanls)/len(ypredrmeanls), sample_weight = weights)))   
 

'''
Epoch 1 LLoss 0.07719 RmeanLLoss 0.07233 LLoss Avg 0.07719 Rmean Avg 0.07233
Epoch 2 LLoss 0.07427 RmeanLLoss 0.06979 LLoss Avg 0.07199 Rmean Avg 0.06896
Epoch 3 LLoss 0.07323 RmeanLLoss 0.06732 LLoss Avg 0.06965 Rmean Avg 0.06690
Epoch 4 LLoss 0.07191 RmeanLLoss 0.06686 LLoss Avg 0.06829 Rmean Avg 0.06576
Epoch 5 LLoss 0.07473 RmeanLLoss 0.06854 LLoss Avg 0.06758 Rmean Avg 0.06520
Epoch 6 LLoss 0.07560 RmeanLLoss 0.06861 LLoss Avg 0.06692 Rmean Avg 0.06464
Epoch 7 LLoss 0.07335 RmeanLLoss 0.06680 LLoss Avg 0.06623 Rmean Avg 0.06405
'''                               