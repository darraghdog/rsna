N_GPU=1
FOLD=0
SIZE='480'

for WDIR in 'resnext101v01' 'resnext101v02'
do
    for GEPOCH in 0 1 2 3 4  
    do
        for FOLD in  0 1 2
        do
            python3 trainlstm.py  \
                --logmsg Rsna-lstm-$GEPOCH-$FOLD-fp16 --epochs 12 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath scripts/$WDIR  \
                --ttahflip T --ttatranspose T  --lrgamma 0.95 --nbags 12 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048
        done
    done
done

