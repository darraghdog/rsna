from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import os, sys
import numpy as np 
import pandas as pd
from PIL import Image
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch.optim as optim
from collections import OrderedDict

from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

import cv2
import gc
import random
import logging
import datetime

import torchvision
from torchvision import transforms as T
from torchvision.models.resnet import ResNet, Bottleneck
import pretrainedmodels

from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize
                           )

from tqdm import tqdm
from apex import amp

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier

import warnings
warnings.filterwarnings('ignore')

# Print info about environments
parser = optparse.OptionParser()
parser.add_option('-s', '--seed', action="store", dest="seed", help="model seed", default="1234")
parser.add_option('-o', '--fold', action="store", dest="fold", help="Fold for split", default="0")
parser.add_option('-p', '--nbags', action="store", dest="nbags", help="Number of bags for averaging", default="0")
parser.add_option('-e', '--epochs', action="store", dest="epochs", help="epochs", default="5")
parser.add_option('-b', '--batchsize', action="store", dest="batchsize", help="batch size", default="16")
parser.add_option('-r', '--rootpath', action="store", dest="rootpath", help="root directory", default="/share/dhanley2/rsna/")
parser.add_option('-i', '--imgpath', action="store", dest="imgpath", help="root directory", default="data/mount/512X512X6/")
parser.add_option('-w', '--workpath', action="store", dest="workpath", help="Working path", default="densenetv1/weights")
parser.add_option('-f', '--weightsname', action="store", dest="weightsname", help="Weights file name", default="pytorch_model.bin")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="0.00005")
parser.add_option('-g', '--logmsg', action="store", dest="logmsg", help="root directory", default="Recursion-pytorch")
parser.add_option('-c', '--size', action="store", dest="size", help="model size", default="512")
parser.add_option('-a', '--infer', action="store", dest="infer", help="root directory", default="TRN")
parser.add_option('-z', '--wtsize', action="store", dest="wtsize", help="model size", default="999")


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
WTSIZE=int(options.wtsize) if int(options.wtsize) != 999 else SIZE
EPOCHS = int(options.epochs)
n_epochs = EPOCHS 
lr=float(options.lr)
batch_size = int(options.batchsize)
ROOT = options.rootpath
path_data = os.path.join(ROOT, 'data')
path_img = os.path.join(ROOT, options.imgpath)
WORK_DIR = os.path.join(ROOT, options.workpath)
WEIGHTS_NAME = options.weightsname
fold = int(options.fold)
INFER=options.infer

#classes = 1109
device = 'cuda'
print('Data path : {}'.format(path_data))
print('Image path : {}'.format(path_img))

os.environ["TORCH_HOME"] = os.path.join( path_data, 'mount')
logger.info(os.system('$TORCH_HOME'))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    cols = np.where(np.max(flatImage, 1) > threshold)[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    #logger.info(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout

def autocropmin(image, threshold=0, kernsel_size = 10):
        
    img = image.copy()
    SIZE = img.shape[0]
    imgfilt = ndimage.minimum_filter(img.max((2)), size=kernsel_size)
    rows = np.where(np.max(imgfilt, 0) > threshold)[0]
    cols = np.where(np.max(imgfilt, 1) > threshold)[0]
    row1, row2 = rows[0], rows[-1] + 1
    col1, col2 = cols[0], cols[-1] + 1
    row1, col1 = max(0, row1-kernsel_size), max(0, col1-kernsel_size)
    row2, col2 = min(SIZE, row2+kernsel_size), min(SIZE, col2+kernsel_size)
    image = image[col1: col2, row1: row2]
    #logger.info(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout


class IntracranialDataset(Dataset):

    def __init__(self, df, path, labels, transform=None):
        self.path = path
        self.data = df
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.jpg')
        #img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)   
        img = cv2.imread(img_name)      
        try:
            try:
                img = autocrop(img, threshold=0, kernsel_size = image.shape[0]//15)
            except:
                img = autocrop(img, threshold=0)  
        except:
            1
            # logger.info('Problem : {}'.format(img_name))      
        img = cv2.resize(img,(SIZE,SIZE))
        #img = np.expand_dims(img, -1)
        if self.transform:       
            augmented = self.transform(image=img)
            img = augmented['image']   

        #logger.info('Mean {:.5f} Min {:.5f} Max {:.5f}'.format(  img.mean().item(), img.max().item(), img.min().item()))
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, label_cols])
            return {'image': img, 'labels': labels}    
        else:      
            return {'image': img}

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
# Data loaders
dir_train_img = os.path.join(path_data, 'stage_1_train_images_jpg')
dir_test_img = os.path.join(path_data, 'stage_1_test_images_jpg')
dir_train_img = os.path.join(path_data, '/data/sdsml_prod/projects/data/ldc/rsna/data/proc') #   'mount1/proc')
dir_test_img = os.path.join(path_data, '/data/sdsml_prod/projects/data/ldc/rsna/data/proc') #  'mount1/proc')
dir_train_img = os.path.join(path_data, 'proc')
dir_test_img = os.path.join(path_data, 'proc')

# Parameters
n_classes = 6
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

train = pd.read_csv(os.path.join(path_data, 'train.csv.gz'))
logger.info(train.shape)
test = pd.read_csv(os.path.join(path_data, 'test.csv.gz'))
png = glob.glob(os.path.join(dir_train_img, '*.jpg'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)
trnimages = set(train.Image.tolist())
print(len(trnimages))
png = [p for p in png if p in trnimages ]
train = train.set_index('Image').loc[png].reset_index()
logger.info(train.shape)

# get fold
valdf = train[train['fold']==fold].reset_index(drop=True)
trndf = train[train['fold']!=fold].reset_index(drop=True)
    
# Data loaders
mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]
transform_train = Compose([
    #ShiftScaleRotate(),
    #CenterCrop(height = SIZE//10, width = SIZE//10, p=0.3),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                         rotate_limit=20, p=0.3, border_mode = cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

transform_test= Compose([
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_train, labels=True)
valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=False)
tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False)

num_workers = 16
trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(valdataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)
tstloader = DataLoader(tstdataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck

'''
torch.hub.list('rwightman/gen-efficientnet-pytorch', force_reload=True)  
model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
model.classifier = torch.nn.Linear(1280, n_classes)
model.to(device)
'''
#model = torch.load(os.path.join(WORK_DIR, '../../checkpoints/resnext101_32x8d_wsl_checkpoint.pth'))

model_func = pretrainedmodels.__dict__['se_resnext101_32x4d']
model = model_func(num_classes=1000, pretrained='imagenet')
#model = torch.load(os.path.join(WORK_DIR, '../../checkpoints/model_se_resnext50_32x4d.bin'))
model.avg_pool = nn.AdaptiveAvgPool2d(1)
model.last_linear = torch.nn.Linear(model.last_linear.in_features, n_classes)
model.to(device)


criterion = torch.nn.BCEWithLogitsLoss()
def criterion(data, targets, criterion = torch.nn.BCEWithLogitsLoss()):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    loss_all = criterion(data, targets)
    loss_any = criterion(data[:,-1:], targets[:,-1:])
    return (loss_all*6 + loss_any*1)/7

plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

for epoch in range(n_epochs):
    logger.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
    logger.info('-' * 10)
    '''
    if epoch < 2:
        input_model_file = 'weights/model_{}_epoch{}_fold{}.bin'.format(WTSIZE, epoch, fold)
        model.load_state_dict(torch.load(input_model_file))
        continue
    '''
    if INFER not in ['TST', 'EMB', 'VAL']:
        for param in model.parameters():
            param.requires_grad = True
        model.train()    
        tr_loss = 0
        for step, batch in enumerate(trnloader):
            if step%1000==0:
                logger.info('Train step {} of {}'.format(step, len(trnloader)))
            inputs = batch["image"] 
            labels = batch["labels"]
            if False:
                if step%20==0:
                    logger.info('Mean {:.5f} Min {:.5f} Max {:.5f}'.format(  inputs.mean().item(), inputs.max().item(), inputs.min().item()))
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            del inputs, labels, outputs
        epoch_loss = tr_loss / len(trnloader)
        logger.info('Training Loss: {:.4f}'.format(epoch_loss))
        for param in model.parameters():
            param.requires_grad = False
        output_model_file = 'weights/model_{}_epoch{}_fold{}.bin'.format(WTSIZE, epoch, fold)
        torch.save(model.state_dict(), output_model_file)
    else:
        del model
        #model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

        model_func = pretrainedmodels.__dict__['se_resnext101_32x4d']
        model = model_func(num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = torch.nn.Linear(model.last_linear.in_features, n_classes)
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
        for param in model.parameters():
            param.requires_grad = False
        input_model_file = 'weights/model_{}_epoch{}_fold{}.bin'.format(WTSIZE, epoch, fold)
        model.load_state_dict(torch.load(input_model_file))
    model.eval()
    logger.info(model.parameters())
    if INFER not in ['EMB', 'NULL', 'TST']:
        valls = []
        for step, batch in enumerate(valloader):
            if step%1000==0:
                logger.info('Val step {} of {}'.format(step, len(valloader)))
            inputs = batch["image"]
            inputs = inputs.to(device, dtype=torch.float)
            out = model(inputs)
            valls.append(torch.sigmoid(out).detach().cpu().numpy())
        weights = ([1, 1, 1, 1, 1, 2] * valdf.shape[0])
        yact = valdf[label_cols].values.flatten()
        ypred = np.concatenate(valls, 0).flatten()
        valloss = log_loss(yact, ypred, sample_weight = weights)
        logger.info('Epoch {} logloss {:.5f}, fold{}'.format(epoch, valloss, fold))
        valpreddf = pd.DataFrame(np.concatenate(valls, 0), columns = label_cols)
        valdf.to_csv('val_act_fold{}.csv.gz'.format(fold), compression='gzip', index = False)
        valpreddf.to_csv('val_pred_sz{}_wt{}_fold{}_epoch{}.csv.gz'.format(SIZE, WTSIZE, fold, epoch), compression='gzip', index = False)
    if INFER == 'TST':
        tstls = []
        for step, batch in enumerate(tstloader):
            if step%1000==0:
                logger.info('Tst step {} of {}'.format(step, len(tstloader)))
            inputs = batch["image"]
            inputs = inputs.to(device, dtype=torch.float)
            out = model(inputs)
            tstls.append(torch.sigmoid(out).detach().cpu().numpy())
        tstpreddf = pd.DataFrame(np.concatenate(tstls, 0), columns = label_cols)
        test.to_csv('tst_act_fold{}.csv.gz'.format(fold), compression='gzip', index = False)
        tstpreddf.to_csv('tst_pred_sz{}_wt{}_fold{}_epoch{}.csv.gz'.format(SIZE, WTSIZE, fold, epoch), compression='gzip', index = False)
    if INFER=='EMB':
        logger.info('Output embeddings epoch {}'.format(epoch))
        logger.info('Train shape {} {}'.format(*trndf.shape))
        logger.info('Valid shape {} {}'.format(*valdf.shape))
        logger.info('Test  shape {} {}'.format(*test.shape)) 
        trndataset = IntracranialDataset(trndf, path=dir_train_img, transform=transform_test, labels=False)
        valdataset = IntracranialDataset(valdf, path=dir_train_img, transform=transform_test, labels=False)
        tstdataset = IntracranialDataset(test, path=dir_test_img, transform=transform_test, labels=False)
        trnloader = DataLoader(trndataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)
        valloader = DataLoader(valdataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)
        tstloader = DataLoader(tstdataset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers)
        # Extract embedding layer
        model.module.last_linear = Identity()
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
        model.eval()
        if epoch < 1:
            continue
        for typ, loader in zip(['tst', 'val', 'trn'], [tstloader, valloader, trnloader]):
            ls = []
            for step, batch in enumerate(loader):
                if step%1000==0:
                    logger.info('Embedding {} step {} of {}'.format(typ, step, len(loader)))
                inputs = batch["image"]
                inputs = inputs.to(device, dtype=torch.float)
                out = model(inputs)
                ls.append(out.detach().cpu().numpy())
                #logger.info('Out shape {}'.format(out.shape))
                #logger.info('Final ls shape {}'.format(ls[-1].shape))
            outemb = np.concatenate(ls, 0)
            logger.info('Write embeddings : shape {} {}'.format(*outemb))
            np.savez_compressed('emb_{}_size{}_fold{}_ep{}'.format(typ, SIZE, fold, epoch), outemb)
            dumpobj('loader_{}_size{}_fold{}_ep{}'.format(typ, SIZE, fold, epoch), loader)
            gc.collect()
