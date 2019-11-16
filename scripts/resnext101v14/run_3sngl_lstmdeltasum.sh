N_GPU=1
WDIR='resnext101v14'
FOLD=0
SIZE='480'
GEPOCH=2

bsub  -q lowpriority  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1827  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainlstmsngl.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 6 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath scripts/$WDIR  \
            --ttahflip F --ttatranspose F  --lrgamma 0.95 --nbags 1 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048 --dropout 0.3 --size $SIZE"

