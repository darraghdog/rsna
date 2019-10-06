import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import zipfile
import os
import io
from PIL import Image
import cv2
import datetime
from tqdm import tqdm

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1, hash_size), Image.ANTIALIAS)
    mat = np.array(
        list(map(lambda x: x[0], image.getdata()))
    ).reshape(hash_size, hash_size+1)
    
    return ''.join(
        map(
            lambda x: hex(x)[2:].rjust(2,'0'),
            np.packbits(np.fliplr(np.diff(mat) < 0))
        )
    )

path = '/Users/dhanley2/Documents/Personal/rsna/data'
trndf = pd.read_csv(os.path.join(path, 'train.csv.gz'))

img_id_hash = []
imgdir = 'stage_1_train_png_224x'
for imname in tqdm(os.listdir(os.path.join(path, imgdir))):
    try:
        img = Image.open(os.path.join(path, imgdir, imname))
        img_hash = dhash(img)
        img_id_hash.append([imname,img_hash])
    except:
        print ('Could not read ' + str(imname))
        
        
        
df = pd.DataFrame(img_id_hash,columns=['Image','image_hash'])
df['Image'] = df['Image'].str.replace('.jpg', '')
#df.to_csv('image_hash_trn.csv')   