### RSNA Intracranial Hemorrhage Detection
  
#### [Hosted on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)  
#### [Sponsored by RSNA](https://www.rsna.org/)   
   
![](https://media.giphy.com/media/WR38jS4CtKttHd7oTU/giphy.gif) 

### Overview    
 
We have a single image classifier (size `480` images with windowing applied), where data is split on 5 folds, but only trained on 3 of them. We then extract the GAP layer (henceforth, we refer to it as the embedding) from the classifier, with TTA, and feed into an LSTM. The above is run with and without preprocessed crop of images; however, just with preprocessed crop achieves same score.

![Alt text](documentation/rsna_nobrainer.png?raw=true "Title")

### Hardware  
    
Ubuntu 16.04 LTS (512 GB boot disk)  
Single Node of 4 x NVIDIA Tesla V100  
16 GB memory per GPU  
4x 1.92 TB SSD RAID 0  
Dual 20-core Intel® Xeon®  
E5-2698 v4 2.2 GHz  

### Software   
Please install docker and run all within a docker environement.   
A docker file is made available `RSNADOCKER.docker` to build.   
Alternatively you can call dockerhub container `darraghdog/kaggle:apex_build`.

### Data set up  
   
1.  Install with `git clone https://github.com/darraghdog/rsna && cd rsna`
2.  Download the raw data and place the zip file `rsna-intracranial-hemorrhage-detection.zip` in subdirectory `./data/raw/`.
3.  Run script `sh run_01_prepare_data.sh` to prepare the meta data and perform image windowing.

This creates the below directory tree.
```
.
├── checkpoints
├── data
│   └── raw
│       ├── stage_2_test_images
│       └── stage_2_train_images
├── docker
├── documentation
├── preds
└── scripts
    ├── resnext101v01
    │   └── weights
    ├── resnext101v02
    │   └── weights
    └── resnext101v03
        └── weights
```
   
### MODEL BUILD: There are three options to produce the solution.  
1) very fast prediction   
    a) runs in 2 hours   
    b) only trains lstm, used pretrained embeddings   
    c) only stage 1 test available for download   
    b) uses precomputed resnext embeddings for a single fold    
2) single run on all training data  
    a) expect this to run for 2 days    
    b) produces single model on all data from scratch       
3) retrain models   
    a) expect this to run about 10 days on a single node   
    b) trains all models from scratch   
    c) makes full bagged submission prediction.
Note: each time you run/rerun one of the above, you should ensure the `/preds` directory is empty.

#### 2. Fast prediction - train lstm only (~2 hours)   

1.  Run script `run_31_fastprediction_only.sh` to download embeddings for a single fold (stage 1 only). This model will achieve a top20 stage 1 result.
           ... if you wish to download stage 2 embeddings run `wget gdown https://drive.google.com/uc?id=1YxCJ0mWIYXfYLN15DPpQ6OLSt4Y54Hp0`    
           ... when you rerun you will need to replace the embeddings/loader with these, and change datapath to `--datapath data` in the lstm run.   
   
#### 2. Retrain single model (2 days)   
    
1.  Run script `run_21_trainsngl_e2e.sh` to train on all data and bag four folds. This model will achieve a top10 result.    

#### 3. Retrain full models (10 days)    
     
1.  Run script `sh run_12_trainfull_imgclassifier.sh` to train the image pipeline.
2.  Run script `sh run_13_trainfull_embedding_extract.sh` to extract image embeddings.
3.  Run script `sh run_14_trainfull_sequential.sh` to train the sequential lstm.
4.  Run script `python scripts/bagged_submission.py` to create bagged submission.

### Insights on what components worked well   

**Preprocessing:**
- Used Appian’s windowing from dicom images. [Linky](https://github.com/darraghdog/rsna/blob/master/eda/window_v1_test.py#L66)
- Cut any black space. There were then headrest or machine artifacts in the image making the head much smaller than it could be - see visual above. These were generally thin lines, so used scipy.ndimage minimum_filter to try to wipe those thin lines. [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainorig.py#L159)
- Albumentations as mentioned in visual above. 

**Image classifier**
- Resnext101 - did not spend a whole lot of time here as it ran so long. But tested SeResenext and Efficitentnetv0 and they did not work as well. 
- Extract GAP layer at inference time  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainorig.py#L387) 

**Create Sequences**
- Extract metadata from dicoms :  [Linky](https://github.com/darraghdog/rsna/blob/master/eda/meta_eda_v1.py) 
- Sequence images on Patient, Study and Series - most sequences were between 24 and 60 images in length.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L200) 

**LSTM**
- Feed in the embeddings in sequence on above key - Patient, Study and Series - also concat on the deltas between current and previous/next embeddings (<current-previous embedding> and <current-next embedding>) to give the model knowledge of changes around the image.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L133) 
- LSTM architecture lifted from the winners of first stage toxic competition. This is a beast - only improvements came from making the hiddens layers larger. Oh, we added on the embeddings to the lstm output and this helped a bit also.  [Linky](https://github.com/darraghdog/rsna/blob/a97018a7b7ec920425189c7e37c1128dd9cb0158/scripts/resnext101v12/trainlstmdeltasum.py#L352) 
- For sequences of different length, padded them to same length, made a dummy embedding of zeros, and then through the results of this away before calculating loss and saving the predictions.  

**What did not help...**  
Too long to do justice... mixup on image, mixup on embedding, augmentations on sequences (partial sequences, reversed sequences), 1d convolutions for sequences (although SeuTao got it working)

**Given more time**  
Make the classifier and the lstm model single end-to-end model. 
Train all on stage2 data, we only got to train two folds of the image model on stage-2 data.
   
