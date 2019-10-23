import os
import pickle
import random
import glob
from PIL import Image


import pandas as pd
import numpy as np
import torch
import cv2
import pydicom
from tqdm import tqdm
from joblib import delayed, Parallel

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


def dumpobj(file, obj):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def loadobj(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def get_dicom_meta(dicom):
    return {
        'PatientID': dicom.PatientID, # can be grouped (20-548)
        'StudyInstanceUID': dicom.StudyInstanceUID, # can be grouped (20-60)
        'SeriesInstanceUID': dicom.SeriesInstanceUID, # can be grouped (20-60)
        'WindowWidth': get_dicom_value(dicom.WindowWidth),
        'WindowCenter': get_dicom_value(dicom.WindowCenter),
        'RescaleIntercept': float(dicom.RescaleIntercept),
        'RescaleSlope': float(dicom.RescaleSlope), # all same (1.0)
    }


def apply_window_policy(image):

    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return image


def apply_dataset_policy(df, policy):
    if policy == 'all':
        pass
    elif policy == 'pos==neg':
        df_positive = df[df.labels != '']
        df_negative = df[df.labels == '']
        df_sampled = df_negative.sample(len(df_positive))
        df = pd.concat([df_positive, df_sampled], sort=False)
    else:
        raise
    log('applied dataset_policy %s (%d records)' % (policy, len(df)))

    return df

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    flatImage = np.max(image, 2)
    rows = np.where(np.max(flatImage, 0) > flatImage.min())[0]
    cols = np.where(np.max(flatImage, 1) > flatImage.min())[0]
    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    sqside = max(image.shape)
    minchl = np.min(image, axis=(0,1))
    imageout = np.ones((sqside, sqside, 3)) * minchl
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout

def convert_dicom_to_npz(imfile):
    try:
        imgnm = (imfile.split('/')[-1]).replace('.dcm', '')
        dicom = pydicom.dcmread(os.path.join(imfile))
        image = dicom.pixel_array
        image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
        image = apply_window_policy(image)
        np.savez_compressed(os.path.join(path_proc, imgnm), image)
    except:
        print(imfile)

import SimpleITK   
def convert_sadicom_to_jpg(name):
    try:

        data = f.read(name)
        dicom = pydicom.dcmread(DicomBytesIO(data))
        image = dicom.pixel_array
        rescale_slope, rescale_intercept = int(dicom.RescaleSlope), int(dicom.RescaleIntercept)
        pos = list(map(float, (dicom.ImagePositionPatient)))
        image = rescale_image(image, rescale_slope, rescale_intercept)
        image = apply_window_policy(image)
        image -= image.min((0,1))
        if image.max()!=1.0:
            print('Image max {}'.format(image.max()))
        image = (255*image).astype(np.uint8)
        cv2.imwrite(os.path.join(path_out, dicom.SOPInstanceUID)+'.jpg', image)
        return dicom.SOPInstanceUID, pos
    except:
        print(name)#(dicom.SOPInstanceUID)
        return '',[]
    
path_img = '/Users/dhanley2/Documents/Personal/rsna/data/CQ500'
path_out = '/Users/dhanley2/Documents/Personal/rsna/data/CQ500OUT'

successls = []
 
stop='CQ500CT103 CQ500CT103/Unknown Study/CT Thin Plain/CT000228.dcm'
zipms = glob.glob(path_img+'/*zip')
namesls = []
import zipfile
import PIL
#import pgmagick
#import gdcm
import glymur
from pydicom.filebase import DicomBytesIO

posdict = {}
for zipf in zipms:
    with zipfile.ZipFile(zipf,  "r") as f:
        try:
            for t, name in enumerate(tqdm(f.namelist())):
                data = f.read(name)
                nm, pos = convert_sadicom_to_jpg(name)
                posdict[nm] = pos
        except:
            1
        
quredf = pd.read_csv(os.path.join(path_img, '../qureai-cq500-boxes.csv'))
quredf = quredf[['SOPInstanceUID','labelName']].drop_duplicates()
quredf = quredf.groupby(['SOPInstanceUID', 'labelName']).size().unstack(fill_value=0)
quredf = quredf.clip(0,1)
quredf.columns = [c.lower() for c in quredf.columns]
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
quredf['any'] = quredf[label_cols[:-1]].max(1)
quredf = quredf[label_cols]
quredf.to_csv(os.path.join(path_img, '../quredf.csv'), compression = 'gzip')