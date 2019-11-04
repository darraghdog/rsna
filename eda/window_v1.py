import os
import pickle
import random
import glob

import pandas as pd
import numpy as np
import torch
import cv2
import pydicom
from tqdm import tqdm
from joblib import delayed, Parallel


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

        
def convert_dicom_to_jpg(name):
    try:
        data = f.read(name)
        dirtype = 'train' if 'train' in name else 'test'
        imgnm = (name.split('/')[-1]).replace('.dcm', '')
        dicom = pydicom.dcmread(DicomBytesIO(data))
        image = dicom.pixel_array
        image = rescale_image(image, rescaledict['RescaleSlope'][imgnm], rescaledict['RescaleIntercept'][imgnm])
        image = apply_window_policy(image)
        image -= image.min((0,1))
        image = (255*image).astype(np.uint8)
        cv2.imwrite(os.path.join(path_proc, dirtype, imgnm)+'.jpg', image)
    except:
        print(name)

path_img = '/Users/dhanley2/Documents/Personal/rsna/data/orig'
BASE_PATH = path_img = '/root/data'
TRAIN_DIR=os.path.join(path_img, 'stage_1_train_images')
TEST_DIR=os.path.join(path_img, 'stage_1_test_images')
path_data = '/Users/dhanley2/Documents/Personal/rsna/data'
path_data = '/root/data/rsna/data'
path_proc = '/Users/dhanley2/Documents/Personal/rsna/data/proc'
path_proc = '/root/data/proc'

trnmdf = pd.read_csv(os.path.join(path_data, 'train_metadata.csv'))
tstmdf = pd.read_csv(os.path.join(path_data, 'test_metadata.csv'))
mdf = pd.concat([trnmdf, tstmdf], 0)
rescaledict = mdf.set_index('SOPInstanceUID')[['RescaleSlope', 'RescaleIntercept']].to_dict()

import zipfile
from pydicom.filebase import DicomBytesIO
with zipfile.ZipFile(os.path.join(path_img, "rsna-intracranial-hemorrhage-detection.zip"), "r") as f:
    for t, name in enumerate(tqdm(f.namelist())):
        convert_dicom_to_jpg(name)

