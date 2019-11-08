N_GPU=1
WDIR='resnext101v13'
FOLD=0
SIZE='480'
GEPOCH=1

for GEPOCH in 0  2 3 4 1  # 5 6 7 8
do
    for FOLD in  0 1 2
    do
        bsub  -q normal  -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1828  -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainlstmdeltasum.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 12 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath scripts/$WDIR  \
            --ttahflip T --ttatranspose F --lrgamma 0.95 --nbags 12 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048 --dropout 0.3 --size $SIZE"
    done
done

