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

def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
    
def f(w3):                        
    def g(x):
        if len(x)>2:
            return (w3*x).mean()
        else:
            return x.mean()
    return g

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
path_data = path='/Users/dhanley2/Documents/Personal/rsna/eda'
wts3 = np.array([0.6, 1.8, 0.6])

# Get image sequences
trnmdf = pd.read_csv(os.path.join(path, '../data/train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path, '../data/test_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
tstmdf = tstmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = (trnmdf.groupby(['PatientID']).cumcount() + 1)
tstmdf['seq'] = (tstmdf.groupby(['PatientID']).cumcount() + 1)
keepcols = ['PatientID', 'SOPInstanceUID', 'seq']
trnmdf = trnmdf[keepcols]
tstmdf = tstmdf[keepcols]
trnmdf.columns = tstmdf.columns = ['PatientID', 'Image', 'seq']

# Load up actuals
trndf = pd.read_csv(os.path.join(path_data, 'seq/v4/trndf.csv.gz'))
valdf = pd.read_csv(os.path.join(path_data, 'seq/v4/valdf.csv.gz'))
tstdf = pd.read_csv(os.path.join(path_data, 'seq/v4/tstdf.csv.gz'))

def makeSub(ypred, imgs):
    imgls = np.array(imgs).repeat(len(label_cols)) 
    icdls = pd.Series(label_cols*ypred.shape[0])   
    yidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    subdf = pd.DataFrame({'ID' : yidx, 'Label': ypred.flatten()})
    return subdf

yactval = makeSub(valdf[label_cols].values, valdf.Image.tolist()).set_index('ID')
ysub = pd.read_csv(os.path.join(path_data, '../sub/lb_sub.csv'), index_col= 'ID')
subbst = pd.read_csv('~/Downloads/sub_pred_sz384_fold5_bag6_wtd_resnextv8.csv.gz', index_col= 'ID')
sublstm = pd.read_csv('~/Downloads/sub_lstm_emb_sz256_wt256_fold0_gepoch235.csv.gz', index_col= 'ID')
ylstmsub = pd.read_csv(os.path.join(path_data, 'seq/v4/lstm_sub_emb_sz256_wt256_fold0_epoch3.csv.gz'), index_col= 'ID')
ylstmval = pd.read_csv(os.path.join(path_data, 'seq/v4/lstm_val_emb_sz256_wt256_fold0_epoch3.csv.gz'), index_col= 'ID')

weights = ([1, 1, 1, 1, 1, 2] * valdf.shape[0])
valloss = log_loss(yactval['Label'].values, ylstmval.loc[yactval.index]['Label'].values, sample_weight = weights)
print('Epoch {} bagged val logloss {:.5f}'.format(3, valloss))


ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
sublstm.Label[sublstm.Label>0.03].hist(bins=100)

idx = trnmdf.Image.isin([i[:12] for i in set(ylstmval.index.tolist())])
set(trnmdf[~idx].PatientID.unique().tolist()).intersection(set(trnmdf[idx].PatientID.unique().tolist()))


ID_46f76e156
ID_59f4dd2a4

trnmdf.Image.apply(len)
print(pd.concat([subbst, ylstmsub], 1).corr())
print(pd.concat([sublstm, ylstmsub], 1).corr())
print(pd.concat([subbst, ysub], 1).corr())









