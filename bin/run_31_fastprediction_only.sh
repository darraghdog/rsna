N_GPU=2
WDIR='resnext101v12fold1'
FOLD=1
SIZE='480'

# Download pretrained embeddings for stage 1 only (25 minutes with fast connection - 16GB file )
pip install gdown
gdown https://drive.google.com/uc?id=13hqPFdCjoMxtAwF863J3Dk33TcBN_wie -O resnext101v12fold1.tar.gz
gunzip resnext101v12fold1.tar.gz
tar -xvf resnext101v12fold1.tar

# Download stage 1 test and train data files
cd resnext101v12fold1/
gdown https://drive.google.com/uc?id=1Fbx3PQHRmJZFc1VNuLKKnnuNUxF1dLe0
gdown https://drive.google.com/uc?id=1XpNW6axRXTfDjLEUD2p48Kro-eZRdO-k
gdown https://drive.google.com/uc?id=15H0b0Ce_3SrvefC22fszekBibGDYEefs
gdown https://drive.google.com/uc?id=1KcF51RnQpSjCBgNzbI4UX2EaUfOS1zHq
cd ../

# Run LSTM for each of the epochs (~1 hour)
for GEPOCH in 0 1 2 3
do            
    python3 scripts/trainlstm.py  \
                --logmsg Rsna-lstm-$GEPOCH-$FOLD-fp16 --epochs 12 --fold $FOLD  --lr 0.00001 --batchsize 4  --workpath $WDIR  \
                --datapath $WDIR --ttahflip F --ttatranspose F  --lrgamma 0.95 --nbags 12 --globalepoch $GEPOCH  --loadcsv F --lstm_units 2048
done

# Create Bagged submission (a minute)
python scripts/bagged_submission.py
