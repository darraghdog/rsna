import sys
import os
import numpy as np
import pandas as pd
import glob
sys.path.insert(0, 'scripts')
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler

logger = get_logger('Make submission', 'INFO') # noqa


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
seqpredsls = glob.glob('preds/lstm*sub*')
logger.info('Load files')
for f in seqpredsls: logger.info(f)
lstmlssub = [pd.read_csv(fname, index_col= 'ID') for fname in seqpredsls]

logger.info('Bag subs')
ylstmsub = sum(lstmlssub)/len(lstmlssub)
ylstmsub = ylstmsub.clip(0.00001, 0.99999)

#ylstmsub.Label[ylstmsub.Label>0.03].hist(bins=100)
logger.info('Create submission')
ylstmsub.to_csv('submission.csv.gz', compression = 'gzip')
