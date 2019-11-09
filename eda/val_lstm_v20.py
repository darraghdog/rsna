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
train2 = pd.read_csv(os.path.join(path_data, '../data/stage_2_train.csv.gz')).set_index('ID')

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
'''
fnamev12 = ['seq/v12/lstmv07/lstmTP2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
fnamev13 = ['seq/v13/lstmv07/lstmT2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
for FOLD,START,BAG,fnamels in zip([0,1,2,0,1],
                              [0]*5,[5]*5,
                              [fnamev12]*3+[fnamev13]*2):
    '''
fnamev12 = ['seq/v12/stg2/lstmv10/lstmTP2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
#fnamev13 = ['seq/v13/lstmv07/lstmT2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
LSTM_UNITS=2048
stg1sub = []
stg2sub = []

for FOLD,START,BAG,fnamels in zip([0,1,2],
                              [0]*3,[5]*3,
                              [fnamev12]*3):
    lstmlssub = []
    lstmlsval = []
    lstmlstst2 = []
    
    for fname in fnamels:
        lstmlssub += [pd.read_csv(os.path.join(path_data, \
                                         fname.format('sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
        lstmlsval += [pd.read_csv(os.path.join(path_data, \
                                         fname.format('val', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
        lstmlstst2 += [pd.read_csv(os.path.join(path_data, \
                                         fname.format('tst2', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
    valdf = train[train.fold==FOLD]
    valdf = valdf[valdf.Image!='ID_9cae9cd5d']
    yactval = makeSub(valdf[label_cols].values, valdf.Image.tolist()).set_index('ID')
    ysub = pd.read_csv(os.path.join(path_data, '../sub/lb_sub.csv'), index_col= 'ID')
    subbst = pd.read_csv('~/Downloads/sub_pred_sz384_fold5_bag6_wtd_resnextv8.csv.gz', index_col= 'ID')
    sublstm = pd.read_csv('~/Downloads/sub_lstm_emb_sz256_wt256_fold0_gepoch235.csv.gz', index_col= 'ID')
    ylstmsub = (sum(lstmlssub)/len(lstmlssub)).clip(0.00001, 0.99999)
    ylstmtst2 = (sum(lstmlstst2)/len(lstmlstst2)).clip(0.00001, 0.99999)

    ylstmval = (sum(lstmlsval)/len(lstmlsval)).clip(0.00001, 0.99999)
    ylstmval = ylstmval[~(pd.Series(ylstmval.index.tolist()).str.contains('ID_9cae9cd5d')).values]
    weights = ([1, 1, 1, 1, 1, 2] * (ylstmval.shape[0]//6))
    ylstmval.loc[yactval.index]['Label'].values
    valloss = log_loss(yactval.loc[ylstmval.index]['Label'].values, ylstmval['Label'].values, sample_weight = weights)
    print('Fold {} Epoch {} Units {} bagged val logloss {:.5f}'.format(FOLD, BAG, LSTM_UNITS, valloss))
    weights = ([1, 1, 1, 1, 1, 2] * (ylstmsub.shape[0]//6))
    subloss = log_loss(train2.loc[ylstmsub.index]['Label'].values, ylstmsub['Label'].values, sample_weight = weights)
    print('Fold {} Epoch {} Units {} bagged stg1 logloss {:.5f}'.format(FOLD, BAG, LSTM_UNITS, subloss))
    stg1sub.append(ylstmsub)
    stg2sub.append(ylstmtst2)


pd.concat(stg1sub, 1).corr()
pd.concat(stg2sub, 1).corr()

ylstmsubbag = sum(stg1sub)/len(stg1sub)
weights = ([1, 1, 1, 1, 1, 2] * (ylstmsubbag.shape[0]//6))
sublossbag = log_loss(train2.loc[ylstmsubbag.index]['Label'].values, ylstmsubbag['Label'].values, sample_weight = weights)
print('Fold {} Epoch {} Units {} bagged stg1 bag logloss {:.5f}'.format(FOLD, BAG, LSTM_UNITS, sublossbag))


'''
baseline
Fold 0 Epoch 5 Units 2048 bagged val logloss 0.05687
Fold 1 Epoch 5 Units 2048 bagged val logloss 0.05859
Fold 2 Epoch 5 Units 2048 bagged val logloss 0.05651
Fold 0 Epoch 5 Units 2048 bagged val logloss 0.05685
Fold 1 Epoch 5 Units 2048 bagged val logloss 0.05839
'''


lstmlssub = []
lstmlstst2 = []

'''
fnamev12 = ['seq/v12/lstmv07/lstmTP2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
fnamev13 = ['seq/v13/lstmv07/lstmT2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
for FOLD,START,BAG,fnamels in zip([0,1,2,0,1],
                              [0]*5,[5]*5,
                              [fnamev12]*3+[fnamev13]*2):
'''
fnamev12 = ['seq/v12/stg2/lstmv10/lstmTP2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
#fnamev13 = ['seq/v13/lstmv07/lstmT2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
LSTM_UNITS=2048
for FOLD,START,BAG,fnamels in zip([0,1,2],
                              [0]*3,[5]*3,
                              [fnamev12]*3):
    for fname in fnamels:
        lstmlssub += [pd.read_csv(os.path.join(path_data, \
                fname.format('sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]
        lstmlstst2 += [pd.read_csv(os.path.join(path_data, \
                fname.format('tst2', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]

ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)
ylstmtst2 = sum(lstmlstst2)/len(lstmlstst2)
ylstmtst2 = ylstmtst2.clip(0.00001, 0.99999)

ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
ylstmtst2.Label[ylstmtst2.Label>0.03].hist(bins=100)


ylstmtst2.to_csv(os.path.join(path, \
            '../sub/substg2_resnext101v12TP_fold012_gepoch01234_LU_2048.csv.gz'), \
            compression = 'gzip')
    
    ylstmval.Label[ylstmval.Label>0.03].hist(bins=100)
