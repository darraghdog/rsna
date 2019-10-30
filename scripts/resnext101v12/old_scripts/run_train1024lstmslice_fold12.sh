N_GPU=1
WDIR='resnext101v12'
FOLD=0
SIZE='480'

for GEPOCH in  0 # 1 2 3 4  # 0 1 2 3 4 5 6 7 8
do
    #for FOLD in 0 # 0 1 2
    for GAMMA in 0.9 0.8 0.7
    do
        bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1830 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainlstmslice.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 9 --fold $FOLD  --lr 0.00005 --batchsize 4  --workpath scripts/$WDIR  \
            --lrgamma $GAMMA --nbags 8 --globalepoch $GEPOCH  --loadcsv F --lstm_units 1024 --dropout 0.3 --size $SIZE"
    done
done
