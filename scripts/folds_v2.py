
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
'''
from skmultilearn.model_selection import IterativeStratification
'''
path_data = '/home/darragh/rsna/data'
path_data = '/Users/dhanley2/Documents/Personal/rsna/data'

trndf = pd.read_csv(os.path.join(path_data, 'stage_1_train.csv.zip'))
tstdf = pd.read_csv(os.path.join(path_data, 'stage_1_sample_submission.csv.zip'))
trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv.gz'))
trn2mdf = pd.read_csv(os.path.join(path_data, 'train_metadata2.csv'))


trn2df = pd.read_csv(os.path.join(path_data, 'stg2/stage_2_train.csv.gz'))
tst2df = pd.read_csv(os.path.join(path_data, 'stg2/stage_2_sample_submission.csv.gz'))


'''
trndf['Image'].isin(trnmdf.SOPInstanceUID.tolist()).value_counts()
trndf['Image'].isin(trnmdf.SeriesInstanceUID.tolist()).value_counts()
trndf['Image'].isin(trnmdf.StudyInstanceUID.tolist()).value_counts()
trnmdf.iloc[0]
trndf.iloc[0]
'''


# Split train out into row per image and save a sample
trndf[['ID', 'Image', 'Diagnosis']] = trndf['ID'].str.split('_', expand=True)
trndf = trndf[['Image', 'Diagnosis', 'Label']]
trndf.drop_duplicates(inplace=True)
trndf = trndf.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
trndf['Image'] = 'ID_' + trndf['Image']
trndf.head()
trndf.shape
trnmdf.shape
trndf.iloc[0]
trnmdf.iloc[0]
trndf['Image']

# Split train out into row per image and save a sample
trn2df[['ID', 'Image', 'Diagnosis']] = trn2df['ID'].str.split('_', expand=True)
trn2df = trn2df[['Image', 'Diagnosis', 'Label']]
trn2df.drop_duplicates(inplace=True)
trn2df = trn2df.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
trn2df['Image'] = 'ID_' + trn2df['Image']
trn2df.head()
trn2df.shape
trn2mdf.shape
trn2df.iloc[0]
trn2mdf.iloc[0]
trn2df['Image']

# Join Patient and split on it
trndf.head()
trndf = trndf.merge(trnmdf[['SOPInstanceUID', 'PatientID']], left_on='Image', right_on='SOPInstanceUID', how='inner')
trndf = trndf.drop('SOPInstanceUID', 1)

trn2df = trn2df.merge(trn2mdf[['SOPInstanceUID', 'PatientID']], left_on='Image', right_on='SOPInstanceUID', how='inner')
trn2df = trn2df.drop('SOPInstanceUID', 1)

# Create folds
folddf = trndf['PatientID'].reset_index(drop=True).drop_duplicates().reset_index()
folddf['fold'] = (folddf['index'].values)%5
folddf = folddf.drop('index', 1)
trndf = trndf.merge(folddf, on='PatientID',  how='inner')
trndf.head()


# Also prepare the test data
tstdf[['ID','Image','Diagnosis']] = tstdf['ID'].str.split('_', expand=True)
tstdf['Image'] = 'ID_' + tstdf['Image']
tstdf = tstdf[['Image', 'Label']]
tstdf.drop_duplicates(inplace=True)
tstdf.to_csv(os.path.join(path_data, 'test.csv.gz'), index=False, compression = 'gzip')

# Also prepare the test data
tst2df[['ID','Image','Diagnosis']] = tst2df['ID'].str.split('_', expand=True)
tst2df['Image'] = 'ID_' + tst2df['Image']
tst2df = tst2df[['Image', 'Label']]
tst2df.drop_duplicates(inplace=True)
tst2df.to_csv(os.path.join(path_data, 'test2.csv.gz'), index=False, compression = 'gzip')

tst2df['Image'].isin(tstdf['Image']).value_counts()
tst2df['Image'].isin(trndf['Image']).value_counts()


# Some small EDA
trndf.columns
trndf.iloc[:,1:].hist(figsize = (10,10))
trndf['Image'].drop_duplicates().shape[0] == trndf['Image'].shape[0]
trndf.shape
tstdf.shape
trndf.head()
tstdf.head()

# Write out the training file
trndf.shape
trndf.to_csv(os.path.join(path_data, 'train.csv.gz'), index=False, compression = 'gzip')
trn2df.to_csv(os.path.join(path_data, 'train2.csv.gz'), index=False, compression = 'gzip')
