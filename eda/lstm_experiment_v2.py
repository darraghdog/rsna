import numpy as np
import math
import pandas as pd
from sklearn.metrics import log_loss
import ast
import pickle

import csv, gzip, os, sys
import math
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

import logging
from tqdm import tqdm
import datetime
import optparse
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

pd.set_option('display.max_rows', 1000)

ROOT = '/Users/dhanley2/Documents/Personal/rsna'
path_data = os.path.join(ROOT, 'data')
path_emb =  os.path.join(ROOT, 'eda/emb/resnext101v11')
n_classes = 6
SIZE=384
fold=0
batch_size=4
GLOBALEPOCH=0
LSTM_UNITS=32#1024
EPOCHS=9
lr=0.0001 
nbags=8 
DROPOUT=0.3
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']



def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)
    
    
def makeSub(ypred, imgs):
    imgls = np.array(imgs).repeat(len(label_cols)) 
    icdls = pd.Series(label_cols*ypred.shape[0])   
    yidx = ['{}_{}'.format(i,j) for i,j in zip(imgls, icdls)]
    subdf = pd.DataFrame({'ID' : yidx, 'Label': ypred.flatten()})
    return subdf

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
        patdelta = (patemb[1:]-patemb[:-1])
        ids = torch.tensor(patdf['embidx'].values)
        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            return {'emb': patemb, 'embidx' : ids, 'labels': labels}    
        else:      
            return {'emb': patemb, 'embidx' : ids}

def predict(loader):
    valls = []
    imgls = []
    imgdf = loader.dataset.data.reset_index().set_index('embidx')[['Image']].copy()
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
        # Get the list of images
        embidx = batch["embidx"].detach().cpu().numpy().astype(np.int32)
        embidx = embidx.flatten()[embidx.flatten()>-1]
        images = imgdf.loc[embidx].Image.tolist() 
        imgls += images
    return np.concatenate(valls, 0), imgls

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
        b['embidx'] = torch.cat((torch.ones((masklen),dtype=torch.long)*-1, b['embidx']))
        b['mask'] = np.ones((maxlen))
        b['mask'][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack((np.zeros((maxlen-len(b['labels']), labdim)), b['labels']))
            
    outbatch = {'emb' : torch.tensor(np.vstack([np.expand_dims(b['emb'], 0) \
                                                for b in batch])).float()}  
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                                for b in batch])).float()
    outbatch['embidx'] = torch.tensor(np.vstack([np.expand_dims(b['embidx'], 0) \
                                                for b in batch])).float()
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch

# Get image sequences
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
trnmdf['seq'] = (trnmdf.groupby(['PatientID']).cumcount() + 1)
tstmdf['seq'] = (tstmdf.groupby(['PatientID']).cumcount() + 1)

keepcols = ['PatientID', 'SOPInstanceUID', 'seq']#+['ImagePos1_lag', 'ImagePos2_lag', 'ImagePos3_lag']
trnmdf = trnmdf[keepcols]
tstmdf = tstmdf[keepcols]
trnmdf.columns = tstmdf.columns = ['PatientID', 'Image', 'seq']#+['ImagePos1_lag', 'ImagePos2_lag', 'ImagePos3_lag']

# Load Data Frames
trndf = loadobj(os.path.join(path_emb, 'loader_trn_size{}_fold{}_ep{}'.format(SIZE, fold, GLOBALEPOCH))).dataset.data
valdf = loadobj(os.path.join(path_emb, 'loader_val_size{}_fold{}_ep{}'.format(SIZE, fold, GLOBALEPOCH))).dataset.data
tstdf = loadobj(os.path.join(path_emb, 'loader_tst_size{}_fold{}_ep{}'.format(SIZE, fold, GLOBALEPOCH))).dataset.data
trndf['embidx'] = range(trndf.shape[0])
valdf['embidx'] = range(valdf.shape[0])
tstdf['embidx'] = range(tstdf.shape[0])
trndf = trndf.merge(trnmdf.drop('PatientID', 1), on = 'Image')
valdf = valdf.merge(trnmdf.drop('PatientID', 1), on = 'Image')
tstdf = tstdf.merge(tstmdf, on = 'Image')

trnemb = np.load(os.path.join(path_emb, 'emb_trn_size{}_fold{}_ep{}.npz'.format(SIZE, fold, GLOBALEPOCH)))['arr_0']
valemb = np.load(os.path.join(path_emb, 'emb_val_size{}_fold{}_ep{}.npz'.format(SIZE, fold, GLOBALEPOCH)))['arr_0']
tstemb = np.load(os.path.join(path_emb, 'emb_tst_size{}_fold{}_ep{}.npz'.format(SIZE, fold, GLOBALEPOCH)))['arr_0']

print('Trn shape {} {}'.format(*trnemb.shape))
print('Val shape {} {}'.format(*valemb.shape))
print('Tst shape {} {}'.format(*tstemb.shape))

print('Create loaders...')
trndataset = IntracranialDataset(trndf, trnemb, labels=True)
trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collatefn)

valdataset = IntracranialDataset(valdf, valemb, labels=False)
tstdataset = IntracranialDataset(tstdf, tstemb, labels=False)
tstloader = DataLoader(tstdataset, batch_size=batch_size*4, shuffle=False, num_workers=8, collate_fn=collatefn)
valloader = DataLoader(valdataset, batch_size=batch_size*4, shuffle=False, num_workers=8, collate_fn=collatefn)

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
        
        ids = torch.tensor(patdf['embidx'].values)
        if self.labels:
            labels = torch.tensor(patdf[label_cols].values)
            return {'emb': patemb, 'embidx' : ids, 'labels': labels}    
        else:      
            return {'emb': patemb, 'embidx' : ids}

for b in trnloader:
    break

b['emb'].shape
b['emb'][0].shape

a = b['emb'][0][b['emb'][0].numpy().sum(1)>0]

a.min()
a.mean()
pd.Series((a[1:]).numpy().flatten()).hist()

pd.Series((a[1:]- a[:-1]).numpy().flatten()).hist()

np.concatenate((padmat,( a[1:]- a[:-1]).numpy()),0).shape
padmat = np.expand_dims(np.zeros(a.numpy().shape[1]), 0)


# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size=trnemb.shape[-1]+3, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        
        self.embedding_dropout = SpatialDropout(DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear(hidden)
        #output = self.linear(h_lstm1)
        
        return output

model =     NeuralNet(LSTM_UNITS=LSTM_UNITS, DO = DROPOUT)
device = 'cpu'
model = model.to(device)
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
# baseline rmean : 0.06893
# 2019-10-12 18:25:38,787 - SequenceLSTM - INFO - Epoch 0 logloss 0.06586674622676458

ypredls = []
ypredtstls = []

for epoch in range(EPOCHS):
    tr_loss = 0.
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    for step, batch in enumerate(tqdm(trnloader)):
        #if step>100:
        #    break
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
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
        optimizer.step()
        
        if step%1000==0:
            print('Trn step {} of {} trn lossavg {:.5f}'. \
                        format(step, len(trnloader), (tr_loss/(1+step))))
            
    model.eval()
    
    print('Prep val score...')
    ypred, imgval = predict(valloader)
    ypredls.append(ypred)
    yvalpred = sum(ypredls[-nbags:])/len(ypredls[-nbags:])
    yvalout = makeSub(yvalpred, imgval)

    if epoch==EPOCHS-1: yvalout.to_csv(os.path.join(path_emb, 'lstm{}deep_val_{}.csv.gz'.format(LSTM_UNITS, embnm)), \
            index = False, compression = 'gzip')
    
    # get Val score
    weights = ([1, 1, 1, 1, 1, 2] * ypred.shape[0])
    yact = valloader.dataset.data[label_cols].values#.flatten()
    yact = makeSub(yact, valloader.dataset.data['Image'].tolist())
    yact = yact.set_index('ID').loc[yvalout.ID].reset_index()
    vallossavg = log_loss(yact['Label'].values, yvalout['Label'].values, sample_weight = weights)
    print('Epoch {} bagged val logloss {:.5f}'.format(epoch, vallossavg))