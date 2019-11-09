### RSNA Intracranial Hemorrhage Detection
  
[Hosted on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)  
[Sponsored by RSNA](https://www.rsna.org/)   
   
![](https://media.giphy.com/media/WR38jS4CtKttHd7oTU/giphy.gif) 
   
### Steps to reproduce submissions
   
Note: Run environment with Docker file `docker/RSNADOCKER.docker`.    
Note: The scripts below were run on an LSF cluster. This can be run outside of LSF, however within the above docker env, by just running the shell command within the double quotes. For eaxample, instead of `bsub -app gpu -n =0  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build "cd /mydir  && python3 myscript.py"`, just run, `cd /mydir  && python3 myscript.py`.    
   
1.  
    
    A. Run script `download.sh` to download the data to `/data` folder. For pretrained image weights we use `torchvision.models.resnet` and the checkpoint `resnext101_32x8d_wsl_checkpoint.pth`, taken from [here](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/). We also attach for your convenince.  
    B. This should also run a script `eda/window_meta2csv.py` to create meta data files. 
    
2. Create folds by executing `python eda/folds_v2.py`   
3. Convert dicoms to jpeg by executing `eda/window_v1.py`. Before doing this point the variables, `path_img` where the dicom images are stored, `path_data` at the meta file directories and `path_proc` to the directory to store the images which will be loaded to the `trainorig` script.   

4. 
    
    A. Run training of resenext101 for 3 folds for 5 epochs by executing `sh scripts/resnext101v12/run_1final_train480.sh`.     
    B. Run again training of resenext101 for 3 folds for 5 epochs by executing `sh scripts/resnext101v13/run_1final_train480.sh`.   

5.
    A. Extract embeddings for each of these runs (3 folds, 5 epochs) using `sh scripts/resnext101v12/run_2final_emb.sh`. Note, this script uses test time augmentation, and extracts embeddings for the original image, horizontal flip and transpose.   

    B. Extract embeddings for each of these runs (3 folds, 5 epochs) using `sh scripts/resnext101v13/run_2final_emb.sh`. Note, this script uses test time augmentation, and extracts embeddings for the original image, horizontal flip and transpose.   
    
6. Train LSTM on image embeddings by sequencing the images per patient, series and study : `sh scripts/resnext101v12/run_3finalA_lstmdeltasum.sh` and `sh scripts/resnext101v13/run_3final_lstmdeltasum.sh`   
7. Train LSTM on image embeddings by sequencing the images per patient, series and study : `sh scripts/resnext101v12/run_3finalB_lstmdeltasum.sh` and  `sh scripts/resnext101v12/run_3final_lstmdeltasum.sh`.    

8. Bag the results of each of the LSTM runs and create submission using `eda/val_lstm_short.py`. Again, 3 folds 5 epochs, and then LSTM is bagged for the 12 epochs it runs. Here we simply average the results  of all the last LSTM file outputs. **submit2**      
9. For blending results. 

    9.1. Run another LSTM using `LSTM_submit_05.py` via: `sh izuit/LSTM_submit_05_f0.sh`; `sh izuit/LSTM_submit_05_f0.sh` and `sh izuit/LSTM_submit_05_f0.sh`. 

    9.2. Bag results using `LSTM_submit_05_bagging.py`

    9.3. Build final blend using `blend.ipynb` with alpha = .95 (best results for stage1 LB) **submit1**   
    
### Results Stage 2

| Model (`.scripts/` folder) |Image Size|LSTM Epochs|Bag|TTA |Fold|Val     |LB    |Comment                          |
| ---------------|----------|------|---|----|----|--------|------|---------------------------------|
| ResNeXt-101 32x8d (v12&v13) with LSTM |480    |5     |LSTM 12X | v12 0 1 2- hflip transpose; v13 0 - hflip |0 1 2 0 1 2  (v12 v13)   |0.05654, 0.05807, 0.05604, 0.05648, 0.05775, 0.05534 |0.654 | Incl stage 1 test, `resnextv12/run_train1024lstmdeltattasum.sh` `resnextv12/run_train1024lstmdeltattasum.sh`  & `eda/val_lstm_v22.py` |
| ResNeXt-101 32x8d (v12&v13) with LSTM |480    |5     |LSTM 12X | v12 0 1 2- hflip transpose; v13 0 - hflip |0 1 2 0 1 2  (v12 v13)   |0.05699, 0.05866, 0.05642, 0.05696, 0.05844, 0.05588 |0.675 | Excl stage 1 test, `resnextv12/run_train1024lstmdeltattasum.sh` `resnextv12/run_train1024lstmdeltattasum.sh`  & `eda/val_lstm_v21.py` |
| ResNeXt-101 32x8d (v12&v13) with LSTM |480    |5     |LSTM 12X | v12 0 1 2- hflip transpose; v13 0 - hflip |0 1 2 0 1 2  (v12 v13)   |0.05699, 0.05866, 0.05642 |0.713 | Excl stage 1 test, `resnextv12/run_train1024lstmdeltattasum.sh` `resnextv12/run_train1024lstmdeltattasum.sh`  & `eda/val_lstm_v20.py` |
    
### Results Stage 1

| Model (`.scripts/` folder) |Image Size|Epochs|Bag|TTA |Fold|Val     |LB    |Comment                          |
| ---------------|----------|------|---|----|----|--------|------|---------------------------------|
| ResNeXt-101 32x8d (v12&v13) with LSTM |480       |5, 5, 5, 5     |LSTM 12X | v12 0 1 2- hflip transpose; v13 0 - hflip |0 1 2 0 (v13)   |0.05705 0.05866 0.05645 0.05690 |0.057 | Hidden 2048, bag12 epochs, `resnextv12/run_train1024lstmdeltattasum.sh` `resnextv12/run_train1024lstmdeltattasum.sh`  & `eda/val_lstm_v14.py` |
| ResNeXt-101 32x8d (v12&v13) with LSTM |480       |5, 5, 5, 5, 5     |LSTM 12X | v12 0 1 2- hflip transpose; v13 0 1 - hflip |0 1 2 0 (v13)   |0.05687 0.05859 0.05651 0.05685 0.05839 |0.057 | Hidden 2048, bag12 epochs, `resnextv12/run_train1024lstmdeltattasum.sh` `resnextv12/run_train1024lstmdeltattasum.sh`  & `eda/val_lstm_v16.py` |
| ResNeXt-101 32x8d (v12) with LSTM |480       |5, 5, 5     |LSTM 12X | all folds 0-hflip, 1 2 - transpose |0 1 2   |0.05705 0.05866 0.05645 |0.057 | Increase hidden units to 2048, bag12 epochs, `scripts/resnextv12/run_train1024lstmdeltatta.sh`  & `eda/val_lstm_v13.py` , bsize 4 patients | 
| ResNeXt-101 32x8d (v12) with LSTM |480       |5     |9X | HFlip TTA on fold0 only|0 1 2   |0.05730 0.05899 0.05681 |0.057 |Concat delta to prev and delta to next, bag9 epochs, `scripts/resnextv12/trainlstmdelta.py`  & `eda/val_lstm_v11.py` , bsize 4 patients | 
| ResNeXt-101 32x8d (v12) with LSTM |480       |6     |5X |None|0   |0.0574 |0.059 | Concat delta to prev and delta to next, bag4 epochs, `scripts/resnextv12/trainlstmdelta.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v11) with LSTM |384       |5, 5, 6     |5X |None|0, 1, 2   |0.05780, 0.05914, 0.05666 |0.059 | 2X LSTM 1024 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py` & `eda/val_lstm_v9.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v12) with LSTM |480       |5     |5X |None|0   |0.05758 |0.059 | 2X LSTM 1024 hidden units, bag4 epochs, `scripts/resnextv12/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v6) with LSTM |384       |7, 5, 7     |6X, 4X, 6X |None|0, 1, 2   |0.5836, 0.6060, 0.5728 |0.060 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v11) with LSTM |384       |5     |5X |None|0   |0.05780 |0.060 | 2X LSTM 1024 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v6) with LSTM |384       |7     |5X |None|0   |0.05811 |0.061 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv6/trainlstmdeep.py`, bsize 4 patients | 
| SEResNeXt-50 32x8d (v3) with LSTM(1024HU) |448      |4     |3X |None|0, 1, 2, 3  |0.05876, 0.06073, 0.05847, 0.06079 |0.061 | 2X LSTM 1024 hidden units, bag8 epochs, `scripts/resnextv6/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v6) with LSTM |384       |7     |3X |None|0   |0.05844 |0.061 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv6/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v8) with LSTM |384       |7     |6X |None|0   |----- |0.062 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv8/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v4) with LSTM |256       |7     |5X |None|0   |0.06119 |0.064 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv4/trainlstmdeep.py`, bsize 4 patients |    
| ResNeXt-101 32x8d (v4) with LSTM |256       |7     |5X |None|0   |0.06217 |0.065 | LSTM 64 hidden units, bag 5 epochs, `scripts/resnextv4/trainlstm.py`, bsize 4 patients |
| ResNeXt-101 32x8d (v8) |384       |7     |5X |None|5 (all)|----- |0.066 | Weighted `[0.6, 1.8, 0.6]` rolling mean win3, transpose, `submission_v6.py`, bsize 128 |
| ResNeXt-101 32x8d (v8) |384       |7     |4X |None|5 (all)|----- |0.067 | Weighted `[0.6, 1.8, 0.6]` rolling mean win3, transpose, `submission_v6.py`, bsize 128 |
| ResNeXt-101 32x8d (v6) |384       |7     |5X |None|0   |0.06336 |0.068 | Weighted `[0.6, 1.8, 0.6]` rolling mean win3, transpose, `submission_v5.py`, bsize 32 |
| ResNeXt-101 32x8d (v4) |256       |7     |5X |None|0   |0.06489 |0.070 | Weighted `[0.6, 1.8, 0.6]` rolling mean win3, transpose, `submission_v4.py`, bsize 64 |
| ResNeXt-101 32x8d (v4) |256       |7     |5X |None|0   |0.06582 |0.070 |Rolling mean window 3, transpose, `submission_v3.py`, bsize 64|
| ResNeXt-101 32x8d (v4) |256       |4     |3X |None|0   |0.06874 |0.074 |Rolling mean window 3, transpose, `submission_v3.py`, bsize 64 |
| EfficientnetV0 (v8) |256       |6     |3X |None|0   |0.07416 |0.081 |Rolling mean window 3, no transpose, `submission_v2.py`, bsize 64 |
| EfficientnetV0 (v8) |384       |4     |2X |None|0   |0.07661 |0.085 |With transpose augmentation      |
| LSTM on logits from ResNeXt-101 32x8d (v4) |256       |3     |3X |None|0   |0.063 |0.082 | LSTM on sequence of patients logits, bsize 4 patients |
| EfficientnetV0 (v8) |384       |2     |1X |None|0   |0.07931 |0.088 |With transpose augmentation      |
| EfficientnetV0 (v8) |384       |11    |2X |None|0   |0.08330 |0.093 |With transpose augmentation      |
| EfficientnetV0 |224       |4     |2X |None|0   |0.08047 |????  |Without transpose augmentation   |
| EfficientnetV0 |224       |4     |2X |None|0   |0.08267 |????  |With transpose augmentation      |
| EfficientnetV0 |224       |2     |1X |None|0   |0.08519 |????  |With transpose augmentation      |
| EfficientnetV0 |224       |11    |2X |None|0   |0.08607 |????  |With transpose augmentation      |
