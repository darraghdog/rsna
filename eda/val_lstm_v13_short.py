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
    
path ='/Users/dhanley2/Documents/Personal/rsna'
path_data ='/Users/dhanley2/Documents/Personal/rsna/lstmoutput/'

lstmlssub = []
fnamels = ['lstm{}{}delta_{}_emb_sz256_wt256_fold{}_epoch{}.csv.gz']
for LSTM_UNITS in ['2048']:
    for FOLD,START,BAG,TTA in zip([0,1,2],[0]*3,[5,5,5],['TP']*3):
        for fname in fnamels:
            lstmlssub += [pd.read_csv(os.path.join(path_data, \
                fname.format(TTA, LSTM_UNITS, 'sub', FOLD, i)), index_col= 'ID') 
                for i in range(START,BAG)]

ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)

ylstmsub.to_csv(os.path.join(path, 'submission.csv.gz'), compression = 'gzip')