#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:58:29 2019

@author: dhanley2
"""
import torch
import os
import numpy as np
from sklearn.metrics import log_loss
import pandas as pd


np.random.seed(100)
y = np.ones((100, 6))
y[:,0] = 0
y_pred = np.random.random((100, 6))
weights = ([1, 1, 1, 1, 1, 2] * y_pred.shape[0])
valloss = log_loss(y.flatten(), y_pred.flatten(), \
                    sample_weight = weights)
valloss
# 1.0588074


def criterion(data, targets, criterion = torch.nn.BCEWithLogitsLoss()):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    loss_all = criterion(data, targets)
    loss_any = criterion(data[:,-1:], targets[:,-1:])
    return (loss_all*6 + loss_any*1)/7
yt = torch.tensor(y)#, dtype = torch.long)
yt_pred = torch.tensor(y_pred)
criterion(yt_pred, yt)

# tensor(1.0588, dtype=torch.float64)
'''
path = '/Users/dhanley2/Documents/Personal/rsna/sub/fold'
yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv')).iloc[:,1:].values
ypred = pd.read_csv(os.path.join(path, 'val_pred_fold0_epoch0.csv')).values
log_loss(yact, ypred)
'''