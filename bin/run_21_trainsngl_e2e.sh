N_GPU=4
WDIR='resnext101v03'
FOLD=6
SIZE='480'

## Run image classifier for all with cropping (~20 hours)
#bsub  -q lowpriority  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
#            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/submit/rsna/  && python3 scripts/trainorig.py  \
#            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 3  --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
#            --imgpath data/proc/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin --autocrop T"

# Extract embeddings for each epoch - no TTA (~15 hours)
#bsub  -q lowpriority  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
#            -m dbslp1829  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/submit/rsna/  && python3 scripts/trainorig.py  \
#            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 3 --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
#            --hflip F --transpose F --infer EMB --imgpath data/proc/ --size $SIZE \
#            --weightsname weights/model_512_resnext101$FOLD.bin"


## Run LSTM for each of the epochs (~2 hours)
N_GPU=1
# These steps can run in parallel
for GEPOCH in 0 # 1 2 
do  
bsub  -q lowpriority  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/submit/rsna/  && python3 scripts/trainlstm.py  \
            --logmsg Rsna-lstm-$GEPOCH-$FOLD-fp16 --epochs 12 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath scripts/$WDIR  \
            --size $SIZE --ttahflip F --ttatranspose F  --lrgamma 0.95 --nbags 12 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048"
done

## Create Bagged submission
#python scripts/bagged_submission.py
