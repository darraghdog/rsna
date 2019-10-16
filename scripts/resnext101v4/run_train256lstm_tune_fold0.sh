N_GPU=1
WDIR='resnext101v4'
FOLD=0
SIZE='256'

for LR in 0.00002 0.00005 0.0001 0.005 0.001
do
    bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainlstm.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 20 --fold $FOLD  --lr $LR  --batchsize 32  --workpath scripts/$WDIR  \
            --nbags 4 --globalepoch 3  --loadcsv F --lstm_units 256  --dropout 0.3 --size $SIZE"
done
