import os
import cv2
import glob
import numpy as np
from PIL import Image
import pandas as pd
from scipy import ndimage, misc 

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomCrop, Lambda, NoOp, CenterCrop, Resize
                           )
       
         
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

import time
path_img = '/Users/dhanley2/Documents/Personal/rsna/data/stage_1_train_png_224x'
SIZE=384

path = '/Users/dhanley2/Documents/Personal/rsna/data/proc'
imgls = glob.glob(path+'/*')
len(imgls)

from IPython.display import display
'''
for ii in range(300, 1000):
    try:
        img = cv2.imread(imgls[ii])
        img = cv2.resize(img,(SIZE,SIZE))
        Image.fromarray(img)
        i1 = cv2.resize(autocrop(img), (SIZE//2,SIZE//2))
        try:
            i2 = cv2.resize(autocropmin(img, kernsel_size = SIZE//15), (SIZE//2,SIZE//2))
        except:
            print('AA'*100)
            i2 = cv2.resize(autocrop(img), (SIZE//2,SIZE//2))
        im = np.concatenate((i1,i2), 1)
        im = Image.fromarray(im)
        os.system('pkill eog') #if you use GNOME Viewer
        display(im)
    
        im.close()
        time.sleep(0.1)
    except:
        continue
   ''' 
   
mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                         rotate_limit=20, p=0.3, border_mode = cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

from tqdm import tqdm
meanls = []
stdls = []

for tt, imname in enumerate(imgls):
    img = cv2.imread(imname)
    try:
        img = autocropmin(img, threshold=0) 
    except:
        try:
            img = autocropmin(img, threshold=0) 
        except:
            1
    # logger.info('Problem : {}'.format(img_name))      
    img = cv2.resize(img,(SIZE,SIZE))
    #img = np.expand_dims(img, -1)
    augmented = transform_train(image=img)
    img = augmented['image'] 
    meanls.append( img.mean((1,2)).numpy())
    stdls.append( img.std((1,2)).numpy())
    if tt%100==0:
        print(tt, np.mean(meanls, 0), np.mean(stdls, 0))

meanls/len(imgls)
stdls/len(imgls)

