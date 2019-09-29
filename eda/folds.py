#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 16:08:18 2019

@author: darragh
"""
import os
import numpy as np
import pandas as pd

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
tstdf.to_csv(os.path.join(path_data, 'test.csv'), index=False)