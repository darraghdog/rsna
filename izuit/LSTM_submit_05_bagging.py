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
from scipy.stats import hmean
import statistics as s


def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
    
wts3 = np.array([0.1, 2.8, 0.1])
def f(w3):                        
    def g(x):
        if len(x)>2:
            return (w3*x).mean()
        else:
            return x.mean()
    return g

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
path_data = '/home/dmitry/Kaggle/RSNA_IH/git/rsna'
path= ''
wts3 = np.array([0.6, 1.8, 0.6])

train = pd.read_csv(path_data+'/data/train.csv.gz')
test = pd.read_csv(path_data+'/data/test.csv.gz')
train = train.set_index('Image').reset_index()
train = train[train.Image!='ID_9cae9cd5d']

# Load up actuals
# trndf = pd.read_csv(os.path.join(path_data, 'seq/v4/trndf.csv.gz'))
# valdf = pd.read_csv(os.path.join(path_data, 'seq/v4/valdf.csv.gz'))
# tstdf = pd.read_csv(os.path.join(path_data, 'seq/v4/tstdf.csv.gz'))

def makeSub(ypred, imgs):
    imgls = np.array(imgs).repeat(len(label_cols)) 
    icdls = pd.Series(label_cols*ypred.shape[0])   
    yidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    subdf = pd.DataFrame({'ID' : yidx, 'Label': ypred.flatten()})
    return subdf
results_path = '/data/dmitry/RSNA/EMB/resnext101v12'
# Load lstms
from scipy.stats import gmean
def magicF(y, beta):
    return 0.5 * ((2 * np.abs(y - 0.5)) ** beta) * np.sign(y - 0.5) + 0.5

for LSTM_UNITS in ['2048']:
    for FOLD,START,BAG in zip([0,1,2],[0,0,0], [7,5,5]):
        #FOLD=1'lstm{}deep_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz'
        fname = '04_CNNlstm{}deep_{}_emb_sz480_fold{}_epoch{}.csv.gz'
        lstmlssub = [pd.read_csv(os.path.join(results_path, \
                                             fname.format(LSTM_UNITS, 'sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
        lstmlsval = [pd.read_csv(os.path.join(results_path, \
                                             fname.format(LSTM_UNITS, 'val', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
#         print([(x.values.shape, type(x), x.columns) for x in lstmlsval])
        valdf = train[train.fold==FOLD]
        valdf = valdf[valdf.Image!='ID_9cae9cd5d']
        yactval = makeSub(valdf[label_cols].values, valdf.Image.tolist()).set_index('ID')
#         ysub = pd.read_csv(os.path.join(path_data, '../sub/lb_sub.csv'), index_col= 'ID')
#         subbst = pd.read_csv('~/Downloads/sub_pred_sz384_fold5_bag6_wtd_resnextv8.csv.gz', index_col= 'ID')
#         sublstm = pd.read_csv('~/Downloads/sub_lstm_emb_sz256_wt256_fold0_gepoch235.csv.gz', index_col= 'ID')
#         ylstmsub = magicF((sum(lstmlssub)/len(lstmlssub)), 1.05).clip(0.00001, 0.99999)
#         ylstmval = magicF((sum(lstmlsval)/len(lstmlsval)), 1.05).clip(0.00001, 0.99999)
        ylstmsub = (sum(lstmlssub)/len(lstmlssub)).clip(0.00001, 0.99999)
        ylstmval = (sum(lstmlsval)/len(lstmlsval)).clip(0.00001, 0.99999)
#         ylstmval["Label"] = gmean(np.hstack([x.values for x in lstmlsval]), axis = 1).clip(0.00001, 0.99999)
        ylstmval = ylstmval[~(pd.Series(ylstmval.index.tolist()).str.contains('ID_9cae9cd5d')).values]
        weights = ([1, 1, 1, 1, 1, 2] * (ylstmval.shape[0]//6))
#         ylstmval.loc[yactval.index]['Label'].values
        valloss = log_loss(yactval.loc[ylstmval.index]['Label'].values, ylstmval['Label'].values, sample_weight = weights)
        print('Epoch {} bagged val logloss {:.5f}'.format(3, valloss))
        
        
#         [0],[0], [4] - 0.05876
#         [1],[0], [4] - 0.6073
#         [2],[0], [4] - 0.5847
#         [3],[0], [4] - 0.6079
        
lstmlssub = []
fname = '04_CNNlstm{}deep_{}_emb_sz480_fold{}_epoch{}.csv.gz'
for LSTM_UNITS in ['2048']:
    for FOLD,START,BAG in zip([0,1,2],[0,0,0], [7,5,5]):
        lstmlssub += [pd.read_csv(os.path.join(results_path, \
                                         fname.format(LSTM_UNITS, 'sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
ylstmsub = sum(lstmlssub)/len(lstmlssub)
# ylstmsub = magicF(ylstmsub, 1.05).clip(0.00001, 0.99999)

# ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
# ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
# sublstm.Label[sublstm.Label>0.03].hist(bins=100)

# print(pd.concat([subbst, ylstmsub], 1).corr())
# print(pd.concat([sublstm, ylstmsub], 1).corr())
# print(pd.concat([subbst, ysub], 1).corr())

ylstmsub.to_csv(os.path.join(results_path, './sub/submit_05_sub_lstmCNN_emb_resnext101v12_sz480_LU_2048.csv.gz'), \
            compression = 'gzip')

# subtop = pd.read_csv(os.path.join(results_path, 
#                         './sub/sub_lstmdeep_emb_resnextv6_sz384_fold012_gepoch123456_bag6_LU_256_1024.csv.gz'), index_col= 'ID')

# print(pd.concat([subtop, ylstmsub], 1).corr())
# subbag = subtop.copy()
# subbag['Label'] = ( subtop.loc[subbag.index]['Label'] + ylstmsub.loc[subbag.index]['Label'] ) / 2
# subbag.to_csv(os.path.join(path, '../sub/sub_bag_lstm_resnextv6_seresnextv3.csv.gz'), \
#             compression = 'gzip')