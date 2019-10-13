N_GPU=1
WDIR='resnext101v4'
FOLD=0
SIZE='256'

for GEPOCH in 2 3 4 5
do
    bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainlstm.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 4 --fold $FOLD  --lr 0.0001 --batchsize 4  --workpath scripts/$WDIR  \
            --nbags 3 --globalepoch $GEPOCH  --loadcsv F --lstm_units 64 --dropout 0.3 --size $SIZE"
done
