import os
import cv2
import glob
import numpy as np
from PIL import Image
import pandas as pd
from scipy import ndimage, misc        
         
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


from IPython.display import display

for ii in range(60, 300):
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
        time.sleep(1)
    except:
        continue
    
