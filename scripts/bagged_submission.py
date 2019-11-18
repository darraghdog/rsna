import numpy as np
import pandas as pd
import os
import glob

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
seqpredsls = glob.glob('preds/lstm*delta_epoch*sub*')
print('Load files')
lstmlssub = [pd.read_csv(fname, index_col= 'ID') for fname in seqpredsls]

print('Bag subs')
ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)

#ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
print('Create submission')
ylstmsub.to_csv('submission.csv.gz', compression = 'gzip')