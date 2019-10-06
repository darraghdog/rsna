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


bag=2
trnmdf = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1

path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
ypredls = []
ypredrmeanls = []
for i in range(4):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}.csv.gz'.format(i)))
    ypred[['Image', 'PatientID']] =  yact[['Image', 'PatientID']]
    ypred = ypred.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID', how='inner')
    ypred = ypred.sort_values(['PatientID', 'seq'])
    ypredrmean = ypred.groupby('PatientID')[label_cols].rolling(3, center = True, min_periods=1).mean().values
    #ypredrmean = (ypred[label_cols]  * 0.6).values
    #ypredrmean[:-1] += ypred[label_cols][1:]  * 0.2
    #ypredrmean[1:] += ypred[label_cols][:-1]  * 0.2
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
    

    
yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}.csv.gz'.format(i)))
ypred[['any']]
yact = yact[['Image', 'PatientID', 'any']]
yact['any_pred'] = ypred['any'].values
yact.head()
yact = yact.merge(trnmdf[['SOPInstanceUID', 'seq']], left_on='Image', right_on ='SOPInstanceUID')
yact = yact.sort_values(['PatientID', 'seq'])
yact.to_csv('y_act.csv')
yact.groupby('seq')['any'].mean()
yact.groupby('seq')['any'].mean().plot(figsize=(10,10), grid=True)


pred = yact.groupby('PatientID')['any_pred'].rolling(3, center = True, min_periods=1).mean()

log_loss(yact['any'].values, yact['any_pred'].values)
log_loss(yact['any'].values, (pred))

os.getcwd()

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=2
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
for i in range(20):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'val_pred_384_fold0_epoch{}.csv.gz'.format(i)))
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))

path = '/Users/dhanley2/Documents/Personal/rsna/data/' 
trnmdf = pd.read_csv(os.path.join(path, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path, 'test_metadata.csv'))
metadf = pd.concat([trnmdf, tstmdf], 0)
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
metadf[poscols] = pd.DataFrame(metadf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
metadf = metadf.sort_values(['PatientID']+poscols)[['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
metadf[['PatientID_lead', 'SOPInstanceUID_lead']] = metadf[['PatientID', 'SOPInstanceUID']].shift(-1)
metadf[['PatientID_lag', 'SOPInstanceUID_lag']] = metadf[['PatientID', 'SOPInstanceUID']].shift(1)
ixlead = metadf['PatientID'] != metadf['PatientID_lead']
ixlag  = metadf['PatientID'] != metadf['PatientID_lag']
metadf['SOPInstanceUID_lag'][ixlag] = metadf['SOPInstanceUID'][ixlag]
metadf['SOPInstanceUID_lead'][ixlead] = metadf['SOPInstanceUID'][ixlead]
metadf[sorted(metadf.columns)].head(100)
seqdf = metadf.filter(like='SOP').set_index('SOPInstanceUID')

trndf = pd.read_csv(os.path.join(path, 'train.csv.gz'))
imname1 = trndf.loc[100, 'Image']
imname0, imname2 = seqdf.loc[imname1]

ix0 = trndf[trndf['Image']==imname0].index[0]
ix1 = trndf[trndf['Image']==imname1].index[0]
ix2 = trndf[trndf['Image']==imname2].index[0]

trndf
img_id_hash = []
imgdir = 'stage_1_train_png_224x'
for imname in tqdm(os.listdir(os.path.join(path, imgdir))):
    try:
        img = Image.open(os.path.join(path, imgdir, imname))
        img_hash = dhash(img)
        img_id_hash.append([imname,img_hash])
    except:
        print ('Could not read ' + str(imname))


path = '/Users/dhanley2/Documents/Personal/rsna/data/' 
trnmdf = pd.read_csv(os.path.join(path, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path, 'test_metadata.csv'))
trndf = pd.read_csv(os.path.join(path, 'train.csv.gz'))
tstdf = pd.read_csv(os.path.join(path, 'test.csv.gz'))
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols)[['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
tstmdf = tstmdf.sort_values(['PatientID']+poscols)[['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
trnmdf[['PatientID_lead', 'SOPInstanceUID_lead']] = trnmdf[['PatientID', 'SOPInstanceUID']].shift(-1)
trnmdf[['PatientID_lag', 'SOPInstanceUID_lag']] = trnmdf[['PatientID', 'SOPInstanceUID']].shift(1)

trnmdf.filter(regex='PatientID|SOPInstanceUID').head(100)


# Add Image Sequence
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.width = 999

poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
tstmdf[poscols] = pd.DataFrame(tstmdf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
cols = ['ImageOrientationPatient', 'ImagePositionPatient', 'WindowCenter', 'WindowWidth']
trnmdf = trnmdf.merge(trndf[['Image']+label_cols], left_on = 'SOPInstanceUID', right_on='Image')

trnmdf.groupby('PatientID')['ImagePos3'].nunique().value_counts()
trnmdf[poscols+['PatientID']+label_cols].sort_values(['PatientID']+poscols)

trnmdf[:2].transpose()
trnmdf['RescaleIntercept'].value_counts()
trnmdf = trnmdf.merge(trndf[['Image', 'any']], left_on = 'SOPInstanceUID', right_on='Image')
cols = ['ImageOrientationPatient', 'ImagePositionPatient', 'WindowCenter', 'WindowWidth']
trnmdf.set_index('PatientID')[cols].sort_index()

trnmdf[['ImagePos{}'.format(i) for i in range(1, 4)]] = \
        pd.DataFrame(trnmdf['ImagePositionPatient'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
trnmdf
pd.options.display.max_rows = 999
cols = ['PatientID', 'WindowCenter', 'WindowWidth', 'any'] + ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[cols].sort_values(['PatientID']+['ImagePos{}'.format(i) for i in range(1, 4)]).head(999)



path = '/Users/dhanley2/Documents/Personal/rsna/data'
for imname in os.listdir(os.path.join(path, 'samp_images')):
    img = Image.open(os.path.join(path, 'samp_images', imname))
    print(imname[:-4] in trndf.Image)
    patient
    trndf.set_index('Image').loc[imname[:-4]]


    #df.to_csv('image_hash_trn.csv')   