#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:53:27 2019

@author: dhanley2
"""
import numpy as np
import csv, gzip, os, sys
import math
import torch
from torch import nn
import torch.optim as optim
import logging
import datetime
import optparse
import pandas as pd
import os
from sklearn.metrics import log_loss
import ast
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-s', '--seed', action="store", dest="seed", help="model seed", default="1234")
parser.add_option('-o', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-p', '--nbags', action="store", dest="nbags", help="Number of bags for averaging", default="4")
parser.add_option('-e', '--epochs', action="store", dest="epochs", help="epochs", default="10")
parser.add_option('-b', '--batchsize', action="store", dest="batchsize", help="batch size", default="4")
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/rsna/")
parser.add_option('-i', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/512X512X6/")
parser.add_option('-w', '--workpath', action="store", dest="workpath", help="Working path", default="densenetv1/weights")
parser.add_option('-f', '--weightsname', action="store", dest="weightsname", help="Weights file name", default="pytorch_model.bin")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="0.00005")
parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory", default="Recursion-pytorch")
parser.add_option('-c', '--size', action="store", dest="size", help="model size", default="512")
parser.add_option('-a', '--globalepoch', action="store", dest="globalepoch", help="root directory", default="3")
parser.add_option('-n', '--loadcsv', action="store", dest="loadcsv", help="Convert csv embeddings to numpy", default="F")
parser.add_option('-j', '--lstm_units', action="store", dest="lstm_units", help="Lstm units", default="128")
parser.add_option('-d', '--dropout', action="store", dest="dropout", help="LSTM input spatial dropout", default="0.3")

options, args = parser.parse_args()
package_dir = options.rootpath
sys.path.append(package_dir)
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler

options, args = parser.parse_args()
package_dir = options.rootpath
sys.path.append(package_dir)
from logs import get_logger
from utils import dumpobj, loadobj, GradualWarmupScheduler


# Print info about environments
logger = get_logger(options.logmsg, 'INFO') # noqa
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

device=torch.device('cuda')
logger.info('Device : {}'.format(torch.cuda.get_device_name(0)))
logger.info('Cuda available : {}'.format(torch.cuda.is_available()))
n_gpu = torch.cuda.device_count()
logger.info('Cuda n_gpus : {}'.format(n_gpu ))


logger.info('Load params : time {}'.format(datetime.datetime.now().time()))
for (k,v) in options.__dict__.items():
    logger.info('{}{}'.format(k.ljust(20), v))

SEED = int(options.seed)
SIZE = int(options.size)
EPOCHS = int(options.epochs)
GLOBALEPOCH=int(options.globalepoch)
n_epochs = EPOCHS 
lr=float(options.lr)
batch_size = int(options.batchsize)
ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
path_emb = WORK_DIR = os.path.join(ROOT, options.workpath)
WEIGHTS_NAME = options.weightsname
fold = int(options.fold)
LOADCSV= options.loadcsv=='T'
LSTM_UNITS=int(options.lstm_units)
nbags=int(options.nbags)
DROPOUT=float(options.dropout)
n_classes = 6
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def criterion(data, targets, criterion = torch.nn.BCEWithLogitsLoss()):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    loss_all = criterion(data, targets)
    loss_any = criterion(data[:,-1:], targets[:,-1:])
    return (loss_all*6 + loss_any*1)/7

class IntracranialDataset(Dataset):
    def __init__(self, df, mat, labels=label_cols):
        self.data = df
        self.mat = mat
        self.labels = labels
        self.patients = df.PatientID.unique()
        self.data = self.data.set_index('PatientID')

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        
        patidx = self.patients[idx]
        patdf = self.data.loc[patidx].sort_values('seq')
        patemb = self.mat[patdf['embidx'].values]
        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            return {'emb': patemb, 'labels': labels}    
        else:      
            return {'emb': patemb}

def predict(loader):
    valls = []
    for step, batch in enumerate(loader):
        inputs = batch["emb"]
        mask = batch['mask'].to(device, dtype=torch.int)
        inputs = inputs.to(device, dtype=torch.float)
        logits = model(inputs)
        # get the mask for masked labels
        maskidx = mask.view(-1)==1
        # reshape for
        logits = logits.view(-1, n_classes)[maskidx]
        valls.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(valls, 0)


# Print info about environments
logger = get_logger('SequenceLSTM', 'INFO') # noqa
logger.info('Cuda set up : time {}'.format(datetime.datetime.now().time()))

# Get image sequences
trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path_data, 'test_metadata.csv'))
mdf = pd.concat([trnmdf, tstmdf])
poscols = ['ImagePos{}'.format(i) for i in range(1, 4)]
mdf[poscols] = pd.DataFrame(mdf['ImagePositionPatient']\
              .apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist())
mdf = mdf.sort_values(['PatientID']+poscols)\
                [['PatientID', 'SOPInstanceUID']+poscols].reset_index(drop=True)
mdf['seq'] = mdf.groupby(['PatientID']).cumcount() + 1
mdf.rename(columns={'SOPInstanceUID':'Image'}, inplace = True)
mdf = mdf[['Image', 'seq', 'PatientID']]

# Load Data Frames
trndf = pd.read_csv(os.path.join(path_emb, 'trndf.csv.gz'))
valdf = pd.read_csv(os.path.join(path_emb, 'valdf.csv.gz'))
tstdf = pd.read_csv(os.path.join(path_emb, 'tstdf.csv.gz'))
trndf['embidx'] = range(trndf.shape[0])
valdf['embidx'] = range(valdf.shape[0])
tstdf['embidx'] = range(tstdf.shape[0])
trndf = trndf.merge(mdf.drop('PatientID', 1), on = 'Image')
valdf = valdf.merge(mdf.drop('PatientID', 1), on = 'Image')
tstdf = tstdf.merge(mdf, on = 'Image')

# Load embeddings

embnm='emb_sz256_wt256_fold{}_epoch{}'.format(fold, GLOBALEPOCH)
if LOADCSV:
    logger.info('Convert to npy..')
    coltypes = dict((i, np.float32) for i in range(2048))
    trnemb = pd.read_csv(os.path.join(path_emb, 'trn_{}.csv.gz'.format(embnm)), dtype = coltypes).values
    valemb = pd.read_csv(os.path.join(path_emb, 'val_{}.csv.gz'.format(embnm)), dtype = coltypes).values
    tstemb = pd.read_csv(os.path.join(path_emb, 'tst_{}.csv.gz'.format(embnm)), dtype = coltypes).values
    np.savez_compressed(os.path.join(path_emb, 'trn_{}'.format(embnm)), trnemb)
    np.savez_compressed(os.path.join(path_emb, 'val_{}'.format(embnm)), valemb)
    np.savez_compressed(os.path.join(path_emb, 'tst_{}'.format(embnm)), tstemb)

logger.info('Load npy..')
valemb = np.load(os.path.join(path_emb, 'val_{}.npz'.format(embnm)))['arr_0']
tstemb = np.load(os.path.join(path_emb, 'tst_{}.npz'.format(embnm)))['arr_0']
trnemb = np.load(os.path.join(path_emb, 'trn_{}.npz'.format(embnm)))['arr_0']
logger.info('Trn shape {} {}'.format(*trnemb.shape))
logger.info('Val shape {} {}'.format(*valemb.shape))
logger.info('Tst shape {} {}'.format(*tstemb.shape))


# a simple custom collate function, just to show the idea
def collatefn(batch):
    maxlen = max([l['emb'].shape[0] for l in batch])
    embdim = batch[0]['emb'].shape[1]
    withlabel = 'labels' in batch[0]
    if withlabel:
        labdim= batch[0]['labels'].shape[1]
        
    for b in batch:
        masklen = maxlen-len(b['emb'])
        b['emb'] = np.vstack((np.zeros((masklen, embdim)), b['emb']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack((np.zeros((maxlen-len(b['labels']), labdim)), b['labels']))
            
    outbatch = {'emb' : torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                                for b in batch])).float()}  
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                                for b in batch])).float()
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch

logger.info('Create loaders...')
trndataset = IntracranialDataset(trndf, trnemb, labels=True)
trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collatefn)

valdataset = IntracranialDataset(valdf, valemb, labels=False)
tstdataset = IntracranialDataset(tstdf, tstemb, labels=False)
tstloader = DataLoader(tstdataset, batch_size=batch_size*4, shuffle=False, num_workers=0, collate_fn=collatefn)
valloader = DataLoader(valdataset, batch_size=batch_size*4, shuffle=False, num_workers=0, collate_fn=collatefn)



# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size=trnemb.shape[-1], LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embedding_dropout = SpatialDropout(DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x
        
        h_lstm1, _ = self.lstm1(h_embedding)
        output = self.linear(h_lstm1)
        
        return output
    
model =     NeuralNet(LSTM_UNITS=LSTM_UNITS, DO = DROPOUT)
model = model.to(device)
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
# baseline rmean : 0.06893
# 2019-10-12 18:25:38,787 - SequenceLSTM - INFO - Epoch 0 logloss 0.06586674622676458
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

ypredls = []
ypredtstls = []

for epoch in range(EPOCHS):
    tr_loss = 0.
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    for step, batch in enumerate(trnloader):
        y = batch['labels'].to(device, dtype=torch.float)
        mask = batch['mask'].to(device, dtype=torch.int)
        x = batch['emb'].to(device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = torch.autograd.Variable(y)
        logits = model(x).to(device, dtype=torch.float)
        # get the mask for masked labels
        maskidx = mask.view(-1)==1
        # reshape for
        y = y.view(-1, n_classes)[maskidx]
        logits = logits.view(-1, n_classes)[maskidx]
        # Get loss
        loss = criterion(logits, y)
        
        tr_loss += loss.item()
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        
        if step%1000==0:
            logger.info('Trn step {} of {} trn lossavg {:.5f}'. \
                        format(step, len(trnloader), (tr_loss/(1+step))))
    
    model.eval()
    
    ypred = predict(valloader)
    ypredls.append(ypred)
    dumpobj('{}_ypredvallsval.pk'.format(embnm), ypredls)
    dumpobj('{}_valdataset.patients.pk'.format(embnm),  valdataset.patients)
    patseq = pd.DataFrame(valdataset.patients, columns=['PatientID']).reset_index()
    yact = valdf.merge(patseq, on='PatientID').sort_values(['index', 'seq'])[label_cols]
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact.values.flatten()
    valloss = log_loss(yact, ypred.flatten(), sample_weight = weights)
    vallossavg = log_loss(yact, (sum(ypredls[-nbags:])/len(ypredls[-nbags:])).flatten(), sample_weight = weights)
    logger.info('Epoch {} val logloss {:.5f} bagged val logloss {:.5f} \n'.format(epoch, valloss, vallossavg))
    
    ypredtstls.append(predict(tstloader))
    dumpobj('{}_ypredtstlsval.pk'.format(embnm), ypredtstls)
    dumpobj('{}_tstdataset.patients.pk'.format(embnm), tstdataset.patients)
    patseq = pd.DataFrame(tstdataset.patients, columns=['PatientID']).reset_index()
    yact = tstdf.merge(patseq, on='PatientID').sort_values(['index', 'seq'])[['Image']]
    imgls = yact.Image.repeat(len(label_cols)) 
    icdls = pd.Series(label_cols*yact.shape[0])    
    ytstidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    ytstpred = sum(ypredtstls[-nbags:])/len(ypredtstls[-nbags:])
    ytstout = pd.DataFrame({'ID' : ytstidx, 'Label': ytstpred.flatten()})
    ytstout.to_csv(os.path.join(path_emb, 'sub_lstm_{}.csv.gz'.format(embnm)), \
            index = False, compression = 'gzip')
    output_model_file = 'weights/model_lstm_{}.bin'.format(embnm)
    torch.save(model.state_dict(), output_model_file)

