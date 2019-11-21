N_GPU=4
WDIR='resnext101v03'
FOLD=6
SIZE='480'

# Run image classifier for all with cropping (~20 hours)
python3 scripts/trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 3  --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
            --imgpath data/proc/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin --autocrop T



# Extract embeddings for each epoch - no TTA (~15 hours)
python3 scripts/trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 3 --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/$WDIR  \
            --hflip F --transpose F --infer EMB --imgpath data/proc/ --size $SIZE \
            --weightsname weights/model_512_resnext101$FOLD.bin



# Run LSTM for each of the epochs (~2 hours)
N_GPU=1
# These steps can run in parallel
for GEPOCH in 0 1 2 
do 
python3 scripts/trainlstm.py  \
            --logmsg Rsna-lstm-$GEPOCH-$FOLD-fp16 --epochs 12 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath scripts/$WDIR  \
            --size $SIZE --ttahflip F --ttatranspose F  --lrgamma 0.95 --nbags 12 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048
done


# Create Bagged Submission
python3 scripts/bagged_submission.py
