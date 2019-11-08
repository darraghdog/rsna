N_GPU=2
WDIR='resnext101v13'
FOLD=0
SIZE='480'

for FOLD in   2 # 0 1
do  
    for  HFLIP in F T
    do
        bsub  -q normal -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1896 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 5 --fold $FOLD  --lr 0.00002 --batchsize 32  --workpath scripts/$WDIR  \
            --hflip $HFLIP --transpose F --infer EMB --imgpath data/mount/512X512X6/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin"
    done

    #bsub  -q normal -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
    #        -m dbslp1828 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR  && python3 trainorig.py  \
    #        --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 4 --fold $FOLD  --lr 0.00002 --batchsize 32  --workpath scripts/$WDIR  \
    #        --hflip F --transpose T --infer EMB --imgpath data/mount/512X512X6/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin"
done
