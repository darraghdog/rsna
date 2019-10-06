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

bag=4
trnmdf = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1

ypredls = []
ypredrmeanls = []
for i in range(6):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}.csv.gz'.format(i)))
    #ypred = pd.read_csv(os.path.join(path, 'val_pred_sz384_wt256_fold0_epoch{}.csv.gz'.format(i)))
    ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
    ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
    ypred = ypred.sort_values(['PatientID', 'seq'])
    ypredrmean = ypred.groupby('PatientID')[label_cols]\
                    .rolling(3, center = True, min_periods=1)\
                    .mean().values
    ypredrmean = pd.DataFrame(ypredrmean, index = ypred.index.tolist(), columns = label_cols )
    ypred = ypred.sort_index()
    ypredrmean = ypredrmean.sort_index()

    #ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}_samp1.0.csv.gz'.format(i)))
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredrmean = ypredrmean[label_cols].values.flatten()

    ypredls.append(ypred)    
    ypredrmeanls.append(ypredrmean)

    
    print('Ct {} LLoss {:.5f} RmeanLLoss {:.5f} LLoss Avg {:.5f} Rmean Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, ypredrmean, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights), \
        log_loss(yact, sum(ypredrmeanls[-bag:])/bag, sample_weight = weights)))


# Get image sequences
def bayesMean(df, prior=50, seq = 'Sequence1'):
    globmean = df['any'].mean()
    ctvar = df.groupby(seq)['any'].count()
    meanvar = df.groupby(seq)['any'].mean()
    bayesmean = ((ctvar*meanvar)+(globmean*50))/(ctvar+prior)
    bayesmean.name = 'bayesmean'+seq
    return bayesmean.reset_index()

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
trnmdf['seq1'] = trnmdf.groupby(['PatientID']).cumcount() + 1
tstmdf['seq1'] = tstmdf.groupby(['PatientID']).cumcount() + 1
trnmdf['seq2'] = trnmdf[::-1].groupby(['PatientID']).cumcount() + 1
tstmdf['seq2'] = tstmdf[::-1].groupby(['PatientID']).cumcount() + 1
trnseq = trnmdf[['SOPInstanceUID', 'seq1', 'seq2']]
tstseq = tstmdf[['SOPInstanceUID', 'seq1', 'seq2']]
trnseq.columns = tstseq.columns = ['Image', 'Sequence1', 'Sequence2']

trndf = pd.read_csv(os.path.join(path, '../../data/train.csv.gz'))
tstdf = pd.read_csv(os.path.join(path, '../../data/test.csv.gz'))
trndf = trndf.merge(trnseq, on='Image')
tstdf = tstdf.merge(tstseq, on='Image')
trndf = trndf.merge(bayesMean(trndf, prior=50, seq = 'Sequence1'), on = 'Sequence1')
trndf = trndf.merge(bayesMean(trndf, prior=50, seq = 'Sequence2'), on = 'Sequence2')
tstdf = tstdf.merge(bayesMean(trndf, prior=50, seq = 'Sequence1'), on = 'Sequence1')
tstdf = tstdf.merge(bayesMean(trndf, prior=50, seq = 'Sequence2'), on = 'Sequence2')
os.getcwd()

trndf[['PatientID', 'Image', 'Sequence1', 'Sequence2', 'any', 'bayesmeanSequence1',  'bayesmeanSequence2' ]].to_csv('seq.csv', index = False)

import torch
import torch.nn as nn
n_classes=6
torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=True)  
model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
model.classifier = torch.nn.Linear(1280, n_classes)

class NNet(nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        preloaded = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        self.features = preloaded
        self.classifier = torch.nn.Linear(1280+2, n_classes)
        del preloaded
        
    def forward(self, x, s1, s2):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        logger.info(out.shape)
        logger.info(s1.shape)
        torch.cat([out, s1, s2], dim=1)
        out = self.classifier(out)
        return features

model = NNet(n_classes=6)

# Sequence include sequences
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

trnmdf['seq1'] = trnmdf.groupby(['PatientID']).cumcount() + 1
tstmdf['seq1'] = tstmdf.groupby(['PatientID']).cumcount() + 1
trnmdf['seq2'] = trnmdf[::-1].groupby(['PatientID']).cumcount() + 1
tstmdf['seq2'] = tstmdf[::-1].groupby(['PatientID']).cumcount() + 1
trnseq = trnmdf[['SOPInstanceUID', 'seq1', 'seq2']]
tstseq = tstmdf[['SOPInstanceUID', 'seq1', 'seq2']]
trnseq.columns = tstseq.columns = ['Image', 'Sequence1', 'Sequence2']


trndf = trndf.merge(trnseq, on='Image')
valdf = valdf.merge(trnseq, on='Image')
test = test.merge(tstseq, on='Image')
trndf = trndf.merge(bayesMean(trndf, prior=50, seq = 'Sequence1'), on = 'Sequence1')
trndf = trndf.merge(bayesMean(trndf, prior=50, seq = 'Sequence2'), on = 'Sequence2')
valdf = valdf.merge(bayesMean(trndf, prior=50, seq = 'Sequence1'), on = 'Sequence1')
valdf = valdf.merge(bayesMean(trndf, prior=50, seq = 'Sequence2'), on = 'Sequence2')
test = test.merge(bayesMean(trndf, prior=50, seq = 'Sequence1'), on = 'Sequence1')
test = test.merge(bayesMean(trndf, prior=50, seq = 'Sequence2'), on = 'Sequence2')
