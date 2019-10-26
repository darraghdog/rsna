### RSNA Intracranial Hemorrhage Detection
  
[Hosted on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)  
[Sponsored by RSNA](https://www.rsna.org/)   
   
![Frontpage](https://www.researchgate.net/profile/Sandiya_Bindroo/publication/326537078/figure/fig1/AS:650818105663489@1532178536539/Magnetic-resonance-imaging-MRI-of-the-brain-showing-scattered-punctate-infarcts-in-the.png) . 

#### Results

| Model (`.scripts/` folder) |Image Size|Epochs|Bag|TTA |Fold|Val     |LB    |Comment                          |
| ---------------|----------|------|---|----|----|--------|------|---------------------------------|
| ResNeXt-101 32x8d (v11) with LSTM |384       |5, 5, 6     |5X |None|0, 1, 2   |0.05780, 0.05914, 0.05666 |0.059 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py` & `eda/val_lstm_v9.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v6) with LSTM |384       |7, 5, 7     |6X, 4X, 6X |None|0, 1, 2   |0.5836, 0.6060, 0.5728 |0.060 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py`, bsize 4 patients | 
| ResNeXt-101 32x8d (v11) with LSTM |384       |5     |5X |None|0   |0.05780 |0.060 | 2X LSTM 256 hidden units, bag4 epochs, `scripts/resnextv11/trainlstmdeep.py`, bsize 4 patients | 
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

#### Experiment Results
1. Cropping image gives approx 0.04. 
2. Mix up improves about 0.03 on EfficientnetV0, but obviously takes longer to converge. Convergence time on same model about 20 epochs instead of 5 without mixup. 
3. Remove the transpose on augmentation converges faster only, slightly lower score. 
4. Adding logged image sequence, concept to embedding layer, converges faster, no real improvement. 

#### Experiments
1. Best training augmentation so far (I think it just converges a bit faster, with transpose works a bit faster)... [linky](https://github.com/darraghdog/rsna/blob/a3a50331955be5f3443e548e692a29d041d24cfe/scripts/efficientnetb0v7/trainorig.py#L210)
```
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.01, 
                         rotate_limit=30, p=0.7, border_mode = cv2.BORDER_REPLICATE),
    Transpose(0.5)
    ToTensor()
])
```

#### Does not work
1. Downsampling patients with no conditions... Maybe need another go, on public scripts appears to help. 
