N_GPU=1
WDIR='densenetv53'
FOLD=0
SIZE='512'


bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 trainorig.py  \
            --logmsg Rsna-leaderboard-$SIZE-fp16 --epochs 100 --fold $FOLD  --lr 0.00002 --batchsize 16  --workpath scripts/$WDIR  \
            --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_cut_model_512_densenet$FOLD.bin"
