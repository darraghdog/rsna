N_GPU=1
WDIR='seresnext50v03'
FOLD=0
SIZE='480'

for GEPOCH in 0 1 2 3 4 5 #6
do
    for FOLD in 0 #1 2 #0
    do
        python LSTM_submit_05_stg2.py  \
            --rootpath /home/dmitry/Kaggle/RSNA_IH/git/rsna/ \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 12 --fold $FOLD  --lr 0.0001 --batchsize 4  --workpath scripts/$WDIR  \
            --nbags 6 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048 --dropout 0.15 --size $SIZE
    done
done