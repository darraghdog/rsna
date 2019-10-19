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

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'

trnmdf = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1

wts3 = np.array([0.6, 1.8, 0.6])
#wts5 = np.array([0.5, 1., 2., 1., 0.5])
def f(w3):                        
    def g(x):
        if len(x)>2:
            return (w3*x).mean()
        else:
            return x.mean()
    return g

 
# Ct 0 LLoss 0.07719 RmeanLLoss 0.07233 LLoss Avg 0.07719 Rmean Avg 0.07233
ypredls = []
ypredrmeanls = []
for fold in [0]:
    yact = pd.read_csv(os.path.join(path, 'val_act_fold{}.csv.gz'.format(fold )))
    yactf = yact[label_cols].values.flatten()
    for epoch in range(0, 1):
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        ypred = pd.read_csv(os.path.join(path, '../../eda/seq/sev1/val_pred_sz448_wt448_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))

        #ypred = pd.read_csv(os.path.join(path, 'v4/val_pred_256_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        ypred = pd.read_csv(os.path.join(path, 'v6/val_pred_sz384_wt384_fold{}_epoch{}.csv.gz'.format(fold ,epoch)))
        ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
        ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
        ypred = ypred.sort_values(['PatientID', 'seq'])
        ypredrmean = ypred.groupby('PatientID')[label_cols]\
                        .rolling(3, center = True, min_periods=1).apply(f(wts3)).values
        ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
        ypred = ypred.sort_index()
        ypredrmean = ypredrmean.sort_index()
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}_samp1.0.csv.gz'.format(i)))
        weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
        ypred = ypred[label_cols].values.flatten()
        ypredrmean = ypredrmean[label_cols].values.flatten()
        ypredls.append(ypred)    
        ypredrmeanls.append(ypredrmean)
        print('Epoch {} LLoss {:.5f} RmeanLLoss {:.5f} LLoss Avg {:.5f} Rmean Avg {:.5f}'.format(1+epoch, \
            log_loss(yactf, ypred, sample_weight = weights), \
            log_loss(yactf, ypredrmean, sample_weight = weights), \
            log_loss(yactf, sum(ypredls)/len(ypredls), sample_weight = weights), \
            log_loss(yactf, sum(ypredrmeanls)/len(ypredrmeanls), sample_weight = weights)))   

from decimal import Decimal
Decimal(0.00003 - 2.1000000000000002e-05)
        
# Ct 0 LLoss 0.07719 RmeanLLoss 0.07233 LLoss Avg 0.07719 Rmean Avg 0.07233
ypredls = []
fold = 0
bag=100
epochs=7
ypredrmeanls = []
yact = pd.read_csv(os.path.join(path, 'val_act_fold{}.csv.gz'.format(fold )))
yactf = yact[label_cols].values.flatten()
for t, (ep_from, ep_to, dir_)  in enumerate([(1, 7, 'v4'), (1, 4, '')]):
    for i in range(ep_from, ep_to):
        if t==1:
            ypred = pd.read_csv(os.path.join(path, 'val_pred_sz384_wt384_fold0_epoch{}.csv.gz'.format(i)))
        else:
            ypred = pd.read_csv(os.path.join(path, '{}/val_pred_256_fold{}_epoch{}.csv.gz'.format(dir_, fold ,i)))
        ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
        ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
        ypred = ypred.sort_values(['PatientID', 'seq'])
        ypredrmean = ypred.groupby('PatientID')[label_cols]\
                        .rolling(3, center = True, min_periods=1).apply(f(wts3)).values
        ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
        ypred = ypred.sort_index()
        ypredrmean = ypredrmean.sort_index()
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}_samp1.0.csv.gz'.format(i)))
        weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
        ypred = ypred[label_cols].values.flatten()
        ypredrmean = ypredrmean[label_cols].values.flatten()
        ypredls.append(ypred)    
        ypredrmeanls.append(ypredrmean)
        print('Ct {} LLoss {:.5f} RmeanLLoss {:.5f} LLoss Avg {:.5f} Rmean Avg {:.5f}'.format(i, \
            log_loss(yactf, ypred, sample_weight = weights), \
            log_loss(yactf, ypredrmean, sample_weight = weights), \
            log_loss(yactf, sum(ypredls)/len(ypredls), sample_weight = weights), \
            log_loss(yactf, sum(ypredrmeanls)/len(ypredrmeanls), sample_weight = weights)))   
    
# Ct 0 LLoss 0.07719 RmeanLLoss 0.07233 LLoss Avg 0.07719 Rmean Avg 0.07233
ypredls = []
fold = 0
bag=6
epochs=7
ypredrmeanls = []
yact = pd.read_csv(os.path.join(path, 'val_act_fold{}.csv.gz'.format(fold )))
yactf = yact[label_cols].values.flatten()
for ep_from, ep_to, dir_  in [(1, 7, 'v4'), (1, 7, 'v5')]:
    for i in range(ep_from, ep_to):
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_sz384_wt384_fold0_epoch{}.csv.gz'.format(i)))
        ypred = pd.read_csv(os.path.join(path, '{}/val_pred_256_fold{}_epoch{}.csv.gz'.format(dir_, fold ,i)))
        ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
        ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
        ypred = ypred.sort_values(['PatientID', 'seq'])
        ypredrmean = ypred.groupby('PatientID')[label_cols]\
                        .rolling(3, center = True, min_periods=1).apply(f(wts3)).values
        ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
        ypred = ypred.sort_index()
        ypredrmean = ypredrmean.sort_index()
        #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}_samp1.0.csv.gz'.format(i)))
        weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
        ypred = ypred[label_cols].values.flatten()
        ypredrmean = ypredrmean[label_cols].values.flatten()
        ypredls.append(ypred)    
        ypredrmeanls.append(ypredrmean)
        print('Ct {} LLoss {:.5f} RmeanLLoss {:.5f} LLoss Avg {:.5f} Rmean Avg {:.5f}'.format(i, \
            log_loss(yactf, ypred, sample_weight = weights), \
            log_loss(yactf, ypredrmean, sample_weight = weights), \
            log_loss(yactf, sum(ypredls)/len(ypredls), sample_weight = weights), \
            log_loss(yactf, sum(ypredrmeanls)/len(ypredrmeanls), sample_weight = weights)))   


# GBM Model    
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

yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
ytact = pd.read_csv(os.path.join(path, 'tst_act_fold.csv.gz'))

ypred = sum(pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}.csv.gz'.format(i))) \
            for i in range(1,7))/len(range(1,7))
ytpred = sum(pd.read_csv(os.path.join(path, 'tst_pred_256_fold0_epoch{}.csv.gz'.format(i))) \
            for i in range(1,7))/len(range(1,7))
ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
ytpred[['Image']] =  ytact[['Image']]

ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
ytpred = ytpred.merge(tstmdf[['SOPInstanceUID', 'seq', 'PatientID']], left_on='Image', right_on ='SOPInstanceUID', how='inner')

ypred = ypred.sort_values(['PatientID', 'seq'])
ytpred = ytpred.sort_values(['PatientID', 'seq'])

wts3 = np.array([0.6, 1.8, 0.6])
ypredrmean = ypred.groupby('PatientID')[label_cols]\
                    .rolling(3, center = True, min_periods=1)\
                    .apply(f(wts3)).values
ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols ).sort_index()    

for ii in [3,5,7,11]:
    print('Rolling mean {}'.format(ii))
    ypred[['rmeanpred{}_{}'.format(ii, i) for i in label_cols]] = \
                    pd.DataFrame(ypred.groupby('PatientID')[label_cols]\
                    .rolling(ii, center = True, min_periods=1).mean().values\
                    , index = ypred.index.tolist(), columns = label_cols )
    ytpred[['rmeanpred{}_{}'.format(ii, i) for i in label_cols]] = \
                    pd.DataFrame(ytpred.groupby('PatientID')[label_cols]\
                    .rolling(ii, center = True, min_periods=1).mean().values\
                    , index = ytpred.index.tolist(), columns = label_cols )

ypred = ypred.sort_index()
ytpred = ytpred.sort_index()

X = ypred.set_index('PatientID').drop(['Image', 'SOPInstanceUID','seq'], 1).copy()

trnsamp = X.index.unique()[:2000]
trnidx = X.index.isin(trnsamp)

Xtrn, ytrn = X[trnidx].values, yact[label_cols][trnidx].values
Xval, yval = X[~trnidx].values, yact[label_cols][~trnidx].values
Xtst = ytpred.set_index('PatientID').drop(['Image', 'SOPInstanceUID','seq'], 1).copy()
import pandas
import keras as ks
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# create model
model_in = ks.Input(shape=(Xtrn.shape[1],), dtype='float32')
out = ks.layers.Dense(128, activation='relu')(model_in)
out = ks.layers.Dense(32, activation='relu')(out)
out = ks.layers.Dense(6, activation='linear')(out)
model = ks.Model(model_in, out)
model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=1e-4))
def scoreme(p):
    return log_loss(yval.flatten(), p, \
             sample_weight = [1, 1, 1, 1, 1, 2] * y_kpred.shape[0])


predls = []
predtls = []
nbags=10
llbase = log_loss(yval.flatten(),ypredrmean[label_cols][~trnidx].values.flatten(), \
         sample_weight = [1, 1, 1, 1, 1, 2] * yval.shape[0])
llbase = log_loss(yval.flatten(),ypredrmean[label_cols][~trnidx].values.flatten(), \
         sample_weight = [1, 1, 1, 1, 1, 2] * yval.shape[0])
def scoreme(p):
    return log_loss(yval.flatten(), p, \
             sample_weight = [1, 1, 1, 1, 1, 2] * yval.shape[0])
for i in range(500):
    model.fit(x=Xtrn, y=ytrn, batch_size=64, epochs=1, verbose=0)
    y_vpred = model.predict(x=Xval, verbose=0)
    y_tpred = model.predict(x=Xtst, verbose=0)
    
    predls.append(y_vpred.clip(0.0001, .9999).flatten())
    predtls.append(y_tpred.clip(0.0001, .9999).flatten())
    llbag = scoreme(sum(predls[-min(i+1, nbags):])/min(i+1,nbags))
    llep = scoreme(predls[-1])
    print('Epoch {} Logloss : {:.5f} Bagged{} : {:.5f} Rmean {:.5f}'.format(i, llep, nbags, llbag, llbase))

