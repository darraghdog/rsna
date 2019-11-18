N_GPU=4
WDIR='resnext101v01'
FOLD=0
SIZE='480'

# Run with cropping
for FOLD in 0 1 2 
do
    python3 scripts/trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 5 --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
            --imgpath data/proc/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin --autocrop T
done

# Run without cropping
WDIR='resnext101v02'
for FOLD in 0 1 2 
do
    python3 scripts/trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 5 --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
            --imgpath data/proc/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin --autocrop F
done
