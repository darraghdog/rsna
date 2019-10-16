import os
import cv2
import numpy as np
from PIL import Image

path_img = '/Users/dhanley2/Documents/Personal/rsna/data/stage_1_train_png_224x'
SIZE=512


img = cv2.imread(os.path.join(path_img, 'ID_0a0ae315e.png'))
img = cv2.resize(img,(SIZE,SIZE))

Image.fromarray(img)

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

def autocropv2(image, threshold = 0):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    _,thresh = cv2.threshold(flatImage,10,245,cv2.THRESH_BINARY_INV) # Change
    
    # Perform morphological closing
    out = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 1*np.ones((11, 11), dtype=np.uint8))
    
    # Perform dilation to expand the borders of the text to be sure
    BORDER = SIZE//20
    out = cv2.dilate(thresh, 255*np.ones((BORDER, BORDER), dtype=np.uint8))
    imageout = ((out<100)*255).astype(np.uint8)
    
    rows = np.where(np.max(imageout, 0) > threshold)[0]
    cols = np.where(np.max(imageout, 1) > threshold)[0]
    
    start_cols = np.max(cols[0]-BORDER, 0)
    end_cols = np.min(cols[0]+BORDER, image.shape[1])
    start_rows = np.max(rows[0]-BORDER, 0)
    end_col = np.min(rows[0]+BORDER, image.shape[0])
    
    image = image[start_cols: end_cols + 1, start_rows: end_rows + 1]
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    

    return imageout

import numpy as np
import cv2


import cv2
import numpy as np

gray = img.copy()
_,thresh = cv2.threshold(gray,10,245,cv2.THRESH_BINARY_INV) # Change

# Perform morphological closing
out = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 1*np.ones((11, 11), dtype=np.uint8))

# Perform dilation to expand the borders of the text to be sure
out = cv2.dilate(thresh, 255*np.ones((SIZE//20, SIZE//20), dtype=np.uint8))
Image.fromarray(img)
Image.fromarray(out)

Image.fromarray(autocropv2(gray))
