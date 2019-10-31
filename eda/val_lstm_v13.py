import numpy as np
import math
import pandas as pd
import os
from sklearn.metrics import log_loss
import ast
import pickle
from scipy.stats import hmean
import statistics as s
import statistics as s
from scipy.stats import gmean

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
path_data = path='/Users/dhanley2/Documents/Personal/rsna/eda'
wts3 = np.array([0.6, 1.8, 0.6])

train = pd.read_csv(os.path.join(path_data, '../data/train.csv.gz'))
test = pd.read_csv(os.path.join(path_data, '../data/test.csv.gz'))
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


for LSTM_UNITS in ['2048']:
    for FOLD,START,BAG,TTA in zip([0,1,2],[0]*3,[5,5,5],['TP']*3):
        fnamels = ['seq/v12/lstmv03/lstm{}{}delta_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
        #fnamels += ['seq/v12/lstmv03/lstmTP{}{}delta_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
        lstmlssub = []
        lstmlsval = []
        for fname in fnamels:
            lstmlssub += [pd.read_csv(os.path.join(path_data, \
                                             fname.format(TTA, LSTM_UNITS, 'sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
            lstmlsval += [pd.read_csv(os.path.join(path_data, \
                                             fname.format(TTA, LSTM_UNITS, 'val', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
        valdf = train[train.fold==FOLD]
        valdf = valdf[valdf.Image!='ID_9cae9cd5d']
        yactval = makeSub(valdf[label_cols].values, valdf.Image.tolist()).set_index('ID')
        ysub = pd.read_csv(os.path.join(path_data, '../sub/lb_sub.csv'), index_col= 'ID')
        subbst = pd.read_csv('~/Downloads/sub_pred_sz384_fold5_bag6_wtd_resnextv8.csv.gz', index_col= 'ID')
        sublstm = pd.read_csv('~/Downloads/sub_lstm_emb_sz256_wt256_fold0_gepoch235.csv.gz', index_col= 'ID')
        ylstmsub = (sum(lstmlssub)/len(lstmlssub)).clip(0.00001, 0.99999)
        ylstmval = (sum(lstmlsval)/len(lstmlsval)).clip(0.00001, 0.99999)
        ylstmval = ylstmval[~(pd.Series(ylstmval.index.tolist()).str.contains('ID_9cae9cd5d')).values]
        weights = ([1, 1, 1, 1, 1, 2] * (ylstmval.shape[0]//6))
        ylstmval.loc[yactval.index]['Label'].values
        valloss = log_loss(yactval.loc[ylstmval.index]['Label'].values, ylstmval['Label'].values, sample_weight = weights)
        print('Fold {} Epoch {} Units {} bagged val logloss {:.5f}'.format(FOLD, BAG, LSTM_UNITS, valloss))

lstmlssub = []
fnamels = ['seq/v12/lstmv03/lstm{}{}delta_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
for LSTM_UNITS in ['2048']:
    for FOLD,START,BAG,TTA in zip([0,1,2],[0]*3,[5,5,5],['TP']*3):
        for fname in fnamels:
            lstmlssub += [pd.read_csv(os.path.join(path_data, \
                fname.format(TTA, LSTM_UNITS, 'sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]

ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)

ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
sublstm.Label[sublstm.Label>0.03].hist(bins=100)

print(pd.concat([subbst, ylstmsub], 1).corr())
print(pd.concat([sublstm, ylstmsub], 1).corr())
print(pd.concat([subbst, ysub], 1).corr())

ylstmsub.to_csv(os.path.join(path, \
            '../sub/sub_lstmdelta_emb_resnext101v12_sz480_fold012tta_gepoch01234_LU_2048.csv.gz'), \
            compression = 'gzip')