#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:46:40 2019

@author: dhanley2
"""
import os
import pandas as pd
path_emb = '/Users/dhanley2/Documents/Personal/rsna/emb'
fname = 'sub_lstm_emb_sz256_wt256_fold0_epoch{}.csv.gz'

gepoch = [2,3,5]
i = 2
dfs = pd.concat([pd.read_csv(os.path.join(path_emb, fname.format(i))).set_index('ID') for i in gepoch], 1)
dfs.corr()
subdf = dfs.mean(1).reset_index()
subdf.columns = ['ID', 'Label']
subdf.to_csv(os.path.join(path_emb, '../sub/sub_lstm_emb_sz256_wt256_fold0_gepoch[2,3,5].csv.gz'), index = False)

ysub = pd.read_csv(os.path.join(path_emb, '../sub/lb_sub.csv'))
print(pd.concat([ysub.set_index('ID'), subdf.set_index('ID')], 1).corr())