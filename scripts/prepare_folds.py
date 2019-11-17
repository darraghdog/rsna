import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, 'scripts')
from logs import get_logger


DATAPATH = 'data'
trndf = pd.read_csv(os.path.join(DATAPATH, 'raw/stage_2_train.csv.zip'))
tstdf = pd.read_csv(os.path.join(DATAPATH, 'raw/stage_2_sample_submission.csv.zip'))
trnmdf = pd.read_csv(os.path.join(DATAPATH, 'train_metadata.csv'))


logger.info('Splittrain/test into row per image')
trndf[['ID', 'Image', 'Diagnosis']] = trndf['ID'].str.split('_', expand=True)
trndf = trndf[['Image', 'Diagnosis', 'Label']]
trndf.drop_duplicates(inplace=True)
trndf = trndf.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
trndf['Image'] = 'ID_' + trndf['Image']

tstdf[['ID','Image','Diagnosis']] = tstdf['ID'].str.split('_', expand=True)
tstdf['Image'] = 'ID_' + tstdf['Image']
tstdf = tstdf[['Image', 'Label']]
tstdf.drop_duplicates(inplace=True)


logger.info('Join Patient and split on it')
trndf = trndf.merge(trnmdf[['SOPInstanceUID', 'PatientID']], left_on='Image', right_on='SOPInstanceUID', how='inner')
trndf = trndf.drop('SOPInstanceUID', 1)

logger.info('Create folds')
folddf = trndf['PatientID'].reset_index(drop=True).drop_duplicates().reset_index()
folddf['fold'] = (folddf['index'].values)%5
folddf = folddf.drop('index', 1)
trndf = trndf.merge(folddf, on='PatientID',  how='inner')
trndf.head()

logger.info('Write out the files')
trndf.to_csv(os.path.join(DATAPATH, 'train.csv.gz'), index=False, compression = 'gzip')
tstdf.to_csv(os.path.join(DATAPATH, 'test.csv.gz'), index=False, compression = 'gzip')

