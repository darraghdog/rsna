
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
trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv'))


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

# Join Patient and split on it
trndf.head()
trndf = trndf.merge(trnmdf[['SOPInstanceUID', 'PatientID']], left_on='Image', right_on='SOPInstanceUID', how='inner')
trndf = trndf.drop('SOPInstanceUID', 1)

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
