import os, ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wts3 = np.array([0.6, 1.8, 0.6])
#wts5 = np.array([0.5, 1., 2., 1., 0.5])
def f(w3):                        
    def g(x):
        x = x.astype(np.float32)
        if len(x)>2:
            return (w3*x).mean()
        else:
            return x.mean()
    return g

path_data = '/Users/dhanley2/Documents/Personal/rsna/data'


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', \
              'subarachnoid', 'subdural', 'any']

trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path_data, 'test_metadata.csv'))

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
trnmdf['seq2'] = trnmdf[::-1].groupby(['PatientID']).cumcount() + 1

# Sequence the data
trndf = pd.read_csv(os.path.join(path_data, 'train.csv.gz'))
tstdf = pd.read_csv(os.path.join(path_data, 'test.csv.gz'))
filtcols = ['seq1', 'seq2', 'SOPInstanceUID']
trndf = trndf.merge(trnmdf[filtcols], left_on='Image', right_on = 'SOPInstanceUID')
trndf = trndf.sort_values(['PatientID', 'seq1'])

tmpdf = trndf.groupby('PatientID')[label_cols]\
                    .rolling(3, center = True, min_periods=1).apply(f(wts3))
tmpdf.reset_index().set_index('level_1').tail(30)
trndfsm = trndf.copy()
trndfsm[label_cols] = tmpdf[label_cols].values

# resort indices
trndfsm = trndfsm.sort_index()
trndf = trndf.sort_index()
tstdf = tstdf.sort_index()

trndfsm[trndfsm['any']!=trndf['any']]
trndf = pd.read_csv(os.path.join(path_data, 'train.csv.gz'))
trndf.tail()
trndfsm.tail()


# check results
trndfsm.to_csv(os.path.join(path_data, 'train_smooth.csv.gz'), \
               compression='gzip', index = False)