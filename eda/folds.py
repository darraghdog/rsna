#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:08:18 2019

@author: darragh
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

path_data = '/home/darragh/rsna/data'

trndf = pd.read_csv(os.path.join(path_data, 'stage_1_train.csv.zip'))
tstdf = pd.read_csv(os.path.join(path_data, 'stage_1_sample_submission.csv.zip'))

# Split train out into row per image and save a sample
trndf[['ID', 'Image', 'Diagnosis']] = trndf['ID'].str.split('_', expand=True)
trndf = trndf[['Image', 'Diagnosis', 'Label']]
trndf.drop_duplicates(inplace=True)
trndf = trndf.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
trndf['Image'] = 'ID_' + trndf['Image']
trndf.head()

# Also prepare the test data
tstdf[['ID','Image','Diagnosis']] = tstdf['ID'].str.split('_', expand=True)
tstdf['Image'] = 'ID_' + tstdf['Image']
tstdf = tstdf[['Image', 'Label']]
tstdf.drop_duplicates(inplace=True)
tstdf.to_csv(os.path.join(path_data, 'test.csv.gz'), index=False, compression = 'gzip')

# Some small EDA
trndf.columns
trndf.iloc[:,1:].hist(figsize = (10,10))
trndf['Image'].drop_duplicates().shape[0] == trndf['Image'].shape[0]
trndf.shape
tstdf.shape
trndf.head()
tstdf.head()

# http://scikit.ml/api/skmultilearn.model_selection.iterative_stratification.html
k_fold = IterativeStratification(n_splits=4, order=1, random_state=100)
splits = k_fold.split(trndf[['Image']], trndf.iloc[:,1:])
folds = [trndf['Image'].iloc[x].tolist() for (x,y) in splits ]

trndf['fold'] = 0 
for t, f in enumerate(folds):
    trndf['fold'][~trndf.Image.isin(f)] = t
trndf.groupby('fold')[trndf.columns.tolist()[1:-1]].sum()

# Write out the training file
trndf.shape
trndf.to_csv(os.path.join(path_data, 'train.csv.gz'), index=False, compression = 'gzip')
