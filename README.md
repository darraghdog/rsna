### RSNA Intracranial Hemorrhage Detection
  
[Hosted on Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)  
[Sponsored by RSNA](https://www.rsna.org/)   
   
![Frontpage](https://www.researchgate.net/profile/Sandiya_Bindroo/publication/326537078/figure/fig1/AS:650818105663489@1532178536539/Magnetic-resonance-imaging-MRI-of-the-brain-showing-scattered-punctate-infarcts-in-the.png) . 

| Model          |Image Size|Epochs|Bag|TTA |Fold|Val     |LB    |Comment                          |
| ---------------|----------|------|---|----|----|--------|------|---------------------------------|
| EfficientnetV0 |384       |4     |2X |None|0   |0.07661 |0.085 |                                 |
| EfficientnetV0 |384       |2     |1X |None|0   |0.07931 |0.088 |                                 |
| EfficientnetV0 |384       |11    |2X |None|0   |0.08330 |0.093 |                                 |
| EfficientnetV0 |224       |4     |2X |None|0   |0.08267 |????  |                                 |
| EfficientnetV0 |224       |2     |1X |None|0   |0.08519 |????  |                                 |
| EfficientnetV0 |224       |11    |2X |None|0   |0.08607 |????  |                                 |

Experiment Results
1. Mix up improves about 0.02 on EfficientnetV0, but obviously takes longer to converge.

Experiments
1. ...
