### RSNA Intracranial Hemorrhage Detection
  
#### [Hosted on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)  
#### [Sponsored by RSNA](https://www.rsna.org/)   
   
![](https://media.giphy.com/media/WR38jS4CtKttHd7oTU/giphy.gif) 

### Overview    
 
We have a single image classifier (size `480` images with windowing applied), where data is split on 5 folds, but only trained on 3 of them. We then extract the GAP layer (henceforth, we refer to it as the embedding) from the classifier, with TTA, and feed into an LSTM. The above is run with and without preprocessed crop of images; however, just with preprocessed crop achieves same score.

![Alt text](documentation/rsna_nobrainer.png?raw=true "Title")

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
   
### Steps to reproduce full submission
   
Note: Run environment within Docker file `docker/RSNADOCKER.docker`.    
  
1.  Install with `git clone https://github.com/darraghdog/rsna && cd rsna`
2.  Download the raw data and place the zip file `rsna-intracranial-hemorrhage-detection.zip` in subdirectory `./data/raw/`.
3.  Run script `sh run_1_prepare_data.sh` to prepare the meta data and perform image windowing.
4.  Run script `sh run_2_train_imgclassifier.sh` to train the image pipeline.
5.  Run script `sh run_3_train_embedding_extract.sh` to extract image embeddings.
6.  Run script `sh run_4_train_sequential.sh` to train the sequential lstm.
7.  Run script `python scripts/bagged_submission.py` to create bagged submission.