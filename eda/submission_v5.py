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

wts = np.array([0.6, 1.8, 0.6])
def f(w):                        
    def g(x):
        if len(x)>2:
            return (w*x).mean()
        else:
            return x.mean()
    return g

bag=ep_to-ep_from
ep_from = 2
ep_to = 7
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'

# Get image sequences
trnmdf = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path, '../../data/test_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
tstmdf = tstmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1
tstmdf['seq'] = tstmdf.groupby(['PatientID']).cumcount() + 1


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
for i in range(ep_from, ep_to):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'v6/val_pred_sz384_wt384_fold0_epoch{}.csv.gz'.format(i)))
    ypred[['Image']] =  yact[['Image']]
    ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq', 'PatientID']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
    ypred = ypred.sort_values(['PatientID', 'seq'])
    ypredrmean = ypred.groupby('PatientID')[label_cols]\
                    .rolling(3, center = True, min_periods=1)\
                    .apply(f(wts)).values
    ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
    ypred = ypredrmean.sort_index()    
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))
    
ypredls = []
for i in range(ep_from, ep_to):
    yact = pd.read_csv(os.path.join(path, 'tst_act_fold.csv.gz'))
    imgls = yact.Image.repeat(len(label_cols)) 
    icdls = pd.Series(label_cols*yact.shape[0])
    ypredidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    ypred = pd.read_csv(os.path.join(path, 'v6/tst_pred_sz384_wt384_fold0_epoch{}.csv.gz'.format(i)))

    ypred[['Image']] =  yact[['Image']]
    ypred = ypred.merge(tstmdf[['SOPInstanceUID', 'PatientID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
    ypred = ypred.sort_values(['PatientID', 'seq'])
    ypredrmean = ypred.groupby('PatientID')[label_cols]\
                    .rolling(3, center = True, min_periods=1)\
                    .apply(f(wts)).values
    ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
    ypred = ypredrmean.sort_index()    

    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
ysub = pd.read_csv(os.path.join(path, '../lb_sub.csv'))
ylb =  pd.DataFrame({'ID' : ypredidx, 'Label': sum(ypredls[-bag:])/bag})
ylb.set_index('ID').loc[ysub.ID].values
print(pd.concat([ysub.set_index('ID'), ylb.set_index('ID')], 1).corr())
ylb.to_csv(os.path.join(path, '../sub_pred_sz384_fold0_bag{}_wtd_resnextv6.csv.gz'.format(i, bag)), \
            index = False, compression = 'gzip')
ylb.head()
