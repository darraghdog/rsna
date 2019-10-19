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

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
path_data = path='/Users/dhanley2/Documents/Personal/rsna/eda'
wts3 = np.array([0.6, 1.8, 0.6])

train = pd.read_csv(os.path.join(path_data, '../data/train.csv.gz'))
train = train.set_index('Image').reset_index()
train = train[train.Image!='ID_9cae9cd5d']

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

# Load lstms
for FOLD,BAG in zip([0,1,2], [7,5,7]):
    #FOLD=1
    fname = 'seq/v6/lstmdeep_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz'
    lstmlssub = [pd.read_csv(os.path.join(path_data, \
                                         fname.format('sub', FOLD, i)), index_col= 'ID') for i in range(1,BAG)]
    lstmlsval = [pd.read_csv(os.path.join(path_data, \
                                         fname.format('val', FOLD, i)), index_col= 'ID') for i in range(1,BAG)]
    valdf = train[train.fold==FOLD]
    valdf = valdf[valdf.Image!='ID_9cae9cd5d']
    
    yactval = makeSub(valdf[label_cols].values, valdf.Image.tolist()).set_index('ID')
    ysub = pd.read_csv(os.path.join(path_data, '../sub/lb_sub.csv'), index_col= 'ID')
    subbst = pd.read_csv('~/Downloads/sub_pred_sz384_fold5_bag6_wtd_resnextv8.csv.gz', index_col= 'ID')
    sublstm = pd.read_csv('~/Downloads/sub_lstm_emb_sz256_wt256_fold0_gepoch235.csv.gz', index_col= 'ID')
    ylstmsub = sum(lstmlssub)/len(lstmlssub)
    ylstmval = sum(lstmlsval)/len(lstmlsval)
    ylstmsub = ylstmsub.clip(0.0001, 0.9999)
    ylstmval = ylstmval.clip(0.0001, 0.9999)
    
    ylstmval = ylstmval[~(pd.Series(ylstmval.index.tolist()).str.contains('ID_9cae9cd5d')).values]
    
    
    weights = ([1, 1, 1, 1, 1, 2] * (ylstmval.shape[0]//6))
    ylstmval.loc[yactval.index]['Label'].values
    valloss = log_loss(yactval.loc[ylstmval.index]['Label'].values, ylstmval['Label'].values, sample_weight = weights)
    print('Epoch {} bagged val logloss {:.5f}'.format(3, valloss))

lstmlssub = []
for FOLD,BAG in zip([0,1,2], [7,5,7]):
    lstmlssub += [pd.read_csv(os.path.join(path_data, \
                                         fname.format('sub', FOLD, i)), index_col= 'ID') for i in range(1,BAG)]
ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.0001, 0.9999)

ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
sublstm.Label[sublstm.Label>0.03].hist(bins=100)

print(pd.concat([subbst, ylstmsub], 1).corr())
print(pd.concat([sublstm, ylstmsub], 1).corr())
print(pd.concat([subbst, ysub], 1).corr())

ylstmsub.to_csv(os.path.join(path, '../sub/sub_lstmdeep_emb_resnextv6_sz384_fold012_gepoch123456_bag6.csv.gz'), \
            compression = 'gzip')

