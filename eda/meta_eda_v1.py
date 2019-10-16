import numpy as np
import math
import pandas as pd
import os
from sklearn.metrics import log_loss
import ast

def fn(x, n = 6):
    outls = list(map(float, ast.literal_eval(x)))
    outls += (n-len(outls))*[np.nan]
    return outls


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

path='/Users/dhanley2/Documents/Personal/rsna/data/'
df_lbls = pd.read_feather(os.path.join(path, 'labels.fth'))
df_tst = pd.read_feather(os.path.join(path, 'df_tst.fth'))
df_trn = pd.read_feather(os.path.join(path, 'df_trn.fth'))
dfstats = pd.read_csv(os.path.join(path, 'img_stats.csv.gz'))
dfstats['Image'] = dfstats['Image'].str.replace('.jpg', '')
dfstats.columns = ['Image', 'jpgmean', 'jpgstd']
comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
comb = comb.join(dfstats.set_index('Image'), 'SOPInstanceUID')
comb = comb.drop('fname', 1)
assert not len(comb[comb['any'].isna()])

comb.head().T

repr_flds = ['BitsStored','PixelRepresentation']
comb.pivot_table(values=['img_mean','img_max','img_min','PatientID','any'], index=repr_flds,
                   aggfunc={'img_mean':'mean',
                            'img_max':'max',
                            'img_min':'min',
                            'PatientID':'count',
                            'any':'mean'})

repr_flds = ['BitsStored','PixelRepresentation']
comb.pivot_table(values=['jpgmean', 'jpgstd', 'img_mean','img_max','img_min','PatientID','any'], index=repr_flds,
                   aggfunc={'jpgmean':'mean',
                            'jpgstd':'mean',
                            'img_mean':'mean',
                            'img_max':'max',
                            'img_min':'min',
                            'PatientID':'count',
                            'any':'mean'})
    
comb.pivot_table(values=['WindowCenter','WindowWidth', 
                         'RescaleIntercept', 'RescaleSlope'], 
                            index=repr_flds,
                   aggfunc={'mean','max','min','std','median'}).T

comb.pivot_table(values=['img_mean'], 
                            index=repr_flds,
                   aggfunc={'mean','max','min','std','median'}).T
                         
df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')
df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')
df3 = comb.query('BitsStored==16')
dfs = [df1,df2,df3]






trnmdf = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
trndf = pd.read_csv(os.path.join(path, '../../data/train.csv.gz'))

poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
trnmdf[poscols] = pd.DataFrame(trnmdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
pixcols = ['PixelSpacing{}'.format(i) for i in range(1, 3)]
trnmdf[pixcols] = pd.DataFrame(trnmdf['PixelSpacing']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())

orcols = ['ImageOrient{}'.format(i) for i in range(1, 7)]
trnmdf[orcols] = pd.DataFrame(trnmdf['ImageOrientationPatient'].apply(lambda x: fn(x, n=6)).tolist())
trnmdf = trnmdf.sort_values(['PatientID']+poscols).reset_index(drop=True)
trnmdf['seq'] = trnmdf.groupby(['PatientID']).cumcount() + 1

comb = trndf.join(trnmdf.set_index('SOPInstanceUID')\
                  .drop(['PatientID', 'ImagePositionPatient', \
                         'ImageOrientationPatient', 'PixelSpacing'], 1)\
                  , 'Image')
assert not len(comb[comb['any'].isna()])

comb.head().T

repr_flds = ['BitsStored','PixelRepresentation']
comb.pivot_table(values=['img_mean','img_max','img_min','PatientID','any'], index=repr_flds,
                   aggfunc={'img_mean':'mean','img_max':'max','img_min':'min','PatientID':'count','any':'mean'})