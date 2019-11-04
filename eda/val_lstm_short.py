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

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
path_data = path='/Users/dhanley2/Documents/Personal/rsna/eda'


lstmlssub = []
fnamev12 = ['seq/v12/lstmv07/lstmTP2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
fnamev13 = ['seq/v13/lstmv07/lstmT2048delta_epoch10_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
for FOLD,START,BAG,fnamels in zip([0,1,2,0,1,2],
                              [0]*5,[5]*5,
                              [fnamev12]*3+[fnamev13]*3):
    for fname in fnamels:
        lstmlssub += [pd.read_csv(os.path.join(path_data, \
                fname.format('sub', FOLD, i)), index_col= 'ID') for i in range(START,BAG)]

ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)

ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)

ylstmsub.to_csv(os.path.join(path, \
            '../sub/sub_lstmdelta_emb_resnext101v12TP_fold012__resnext101v13T_fold01_gepoch01234_LU_2048.csv.gz'), \
            compression = 'gzip')