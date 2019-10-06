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

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=6
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
for i in range(20):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz')).iloc[~iix.values]
    ypred = pd.read_csv(os.path.join(path, 'val_pred_fold0_epoch{}.csv.gz'.format(i))).iloc[~iix.values]
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=1
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
yact = pd.read_csv(os.path.join(path, '../../data/train.csv.gz'))
yact = yact[yact['fold']==0].reset_index()#.iloc[~iix.values]
yact = yact[label_cols].values.flatten()
for i in range(3):
    ypred = pd.read_csv(os.path.join(path, 'val_pred_384_fold0_epoch{}.csv.gz'.format(i)))#.iloc[~iix.values]
    weights = ([1, 1, 1, 1, 1, 2] * ypred.shape[0])
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))
ypredls = []
for i in range(3):
    ypred = pd.read_csv(os.path.join(path, 'tst_pred_384_fold0_epoch{}.csv.gz'.format(i)))
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
ylb = pd.read_csv(os.path.join(path, '../lb_sub.csv'))
ylb['Pred'] =  sum(ypredls[-bag:])/bag
print(ylb[['Label', 'Pred']].corr())
ysub = ylb[['ID']]
ysub['Label'] = ylb['Pred'].values
ysub.to_csv(os.path.join(path, '../sub_pred_384_fold0_epoch{}_efficientnet.csv.gz'.format(i)), \
            index = False, compression = 'gzip')
