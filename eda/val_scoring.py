#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:53:27 2019

@author: dhanley2
"""
import numpy as np
import math
import pandas as pd
import os
from sklearn.metrics import log_loss

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=2
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
for i in range(2):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'val_pred_256_fold0_epoch{}.csv.gz'.format(i)))
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))
    
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=2
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
for i in range(20):
    yact = pd.read_csv(os.path.join(path, 'val_act_fold0.csv.gz'))
    ypred = pd.read_csv(os.path.join(path, 'val_pred_384_fold0_epoch{}.csv.gz'.format(i)))
    weights = ([1, 1, 1, 1, 1, 2] * yact.shape[0])
    yact = yact[label_cols].values.flatten()
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))


label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
ypredls = []
bag=1
path='/Users/dhanley2/Documents/Personal/rsna/sub/fold'
yact = pd.read_csv(os.path.join(path, '../../data/train.csv.gz'))
yact = yact[yact['fold']==0].reset_index()#.iloc[~iix.values]
yact = yact[label_cols].values.flatten()
for i in range(3):
    ypred = pd.read_csv(os.path.join(path, 'val_pred_384_fold0_epoch{}.csv.gz'.format(i)))#.iloc[~iix.values]
    weights = ([1, 1, 1, 1, 1, 2] * ypred.shape[0])
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
    print('Ct {} LLoss {:.5f} LLoss Avg {:.5f}'.format(i, \
        log_loss(yact, ypred, sample_weight = weights), \
        log_loss(yact, sum(ypredls[-bag:])/bag, sample_weight = weights)))
ypredls = []
for i in range(3):
    ypred = pd.read_csv(os.path.join(path, 'tst_pred_384_fold0_epoch{}.csv.gz'.format(i)))
    ypred = ypred[label_cols].values.flatten()
    ypredls.append(ypred)
ylb = pd.read_csv(os.path.join(path, '../lb_sub.csv'))
ylb['Pred'] =  sum(ypredls[-bag:])/bag
print(ylb[['Label', 'Pred']].corr())
ysub = ylb[['ID']]
ysub['Label'] = ylb['Pred'].values
ysub.to_csv(os.path.join(path, '../sub_pred_384_fold0_epoch{}_efficientnet.csv.gz'.format(i)), \
            index = False, compression = 'gzip')


import pandas as pd
trndf = pd.read_csv(os.path.join(path, '../../data/stage_1_train.csv'))
tstdf = pd.read_csv(os.path.join(path, '../../data/test.csv'))
trnfdf = pd.read_csv(os.path.join(path, '../../data/train.csv.gz'))#.set_index('fold')

trnfdf

metatrn = pd.read_csv(os.path.join(path, '../../data/train_metadata.csv'))
metatrn['fold'] = trnfdf['fold'].values
metatrn.iloc[0]
metatst = pd.read_csv(os.path.join(path, '../../data/test_metadata.csv'))

trndf[['id', 'Patient', 'Diagnosis']] = trndf['ID'].str.split('_', expand = True)
tstdf[['id', 'Patient']] = tstdf['Image'].str.split('_', expand = True)
metatrn[['id', 'Patient']] = metatrn['PatientID'].str.split('_', expand = True)
metatst[['id', 'Patient']] = metatst['PatientID'].str.split('_', expand = True)

metatrn['Patient'].isin(metatst['Patient'].tolist()).value_counts()
ix = metatrn['fold'] == 0 
iix = metatrn['Patient'][ix].isin(metatst['Patient'][~ix].tolist())
iix.value_counts()

trndf['Patient'].isin(tstdf['Patient'].tolist()).value_counts()



metatrn.columns
metatrn['PatientID']


import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck

'''
# Run below, with internet access
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
torch.save(model, 'resnext101_32x8d_wsl_checkpoint.pth')
'''
#model = torch.load(os.path.join(WORK_DIR, '../../checkpoints/resnext101_32x8d_wsl_checkpoint.pth'))
#model.fc = torch.nn.Linear(2048, n_classes)
torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=True)
model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
model.conv_stem.weight = model.conv_stem.weight.sum(dim=1, keepdim=True)
model.conv_stem.in_channels=1

for t,i in enumerate(model.named_parameters()):
    if t>10:
        break
    print(i)


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """
    flatImage = np.max(image, 2)
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[1]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
            
    #         print(img.shape)
        return img
    
path = '/Users/dhanley2/Documents/Personal/rsna/data/' 
trnmdf = pd.read_csv(os.path.join(path, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path, 'test_metadata.csv'))

trnfdf = pd.read_csv(os.path.join(path, 'train.csv.gz'))#.set_index('fold')


tstmdf = pd.read_csv(os.path.join(path, 'test_metadata.csv'))

trnmdf['fold'] = trnfdf.fold.values

trnmdf.head()
trnmdf.iloc[0]

trnmdf['StudyInstanceUID'].value_counts()

(trnmdf['StudyInstanceUID'].isin(tstmdf['StudyInstanceUID'])).value_counts()

(trnmdf['PatientID'][trnmdf.fold==0].isin(trnmdf['PatientID'][trnmdf.fold==1])).value_counts()

(trnmdf['StudyInstanceUID'][trnmdf.fold==0].isin(trnmdf['StudyInstanceUID'][trnmdf.fold==1])).value_counts()

trn

trnmdf.filter(regex='UID|PatientI')[trnmdf.PatientID=='ID_b81a287f']

trnmdf.filter(regex='UID|PatientI')[trnmdf.StudyInstanceUID=='ID_dd37ba3adb']
trnmdf.filter(regex='UID|PatientI')[trnmdf.SOPInstanceUID=='ID_231d901c1']


    
    
import cv2
path = '/Users/dhanley2/Documents/Personal/rsna/data/samp_images'
img_name = os.path.join(path, 'ID_000012eaf.jpg')
img_name = os.path.join(path, 'ID_716e62b8a.jpg')

imgorig = cv2.imread(img_name) 
img = imgorig.copy()
imgt = cv2.transpose(img)
image = imgorig.copy()

img.shape
autocrop(img, threshold=0).sum()
autocrop(img, threshold=0).shape

cv2.imshow('image',autocrop(img, threshold=0))
cv2.waitKey(0)

img.mean()
img.shape
img1 = crop_image_from_gray(img,tol=7)
img1.shape

cv2.imshow('image',img)
cv2.waitKey(0)


img.shape
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) 
img = np.expand_dims(img, -1)
img.shape
SIZE=224
        if (SIZE!=512) :
            img = cv2.resize(img,(SIZE,SIZE))
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']  