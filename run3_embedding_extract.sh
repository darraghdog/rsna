N_GPU=2
FOLD=0
SIZE='480'

# Run for image classifier with crop and without crop
for WDIR in 'resnext101v01' 'resnext101v02'
do
    # Run for 3 folds
    for FOLD in  0 1 2
    do  
        for HFLIP in F T
        do
            # Extract original and flipped image embeddings
            python3 scripts/trainorig.py  \
                --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 6 --fold $FOLD  --lr 0.00002 --batchsize 32  --workpath scripts/$WDIR  \
                --stage2 T --hflip $HFLIP --transpose F --infer EMB --imgpath data/mount/512X512X6/ --size $SIZE \
                --weightsname weights/model_512_resnext101$FOLD.bin
        done
            # Extract transposed image embeddings
            python3 scripts/trainorig.py  \
                --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 6 --fold $FOLD  --lr 0.00002 --batchsize 32  --workpath scripts/$WDIR  \
                --stage2 T --hflip F --transpose T --infer EMB --imgpath data/mount/512X512X6/ --size $SIZE \
                --weightsname weights/model_512_resnext101$FOLD.bin
    done
done
