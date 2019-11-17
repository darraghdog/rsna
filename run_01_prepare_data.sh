# Parameters
# Clone repo https://github.com/darraghdog/rsna and set the location as ROOT directory
ROOT='/mnt/lsf/share/dhanley2/rsna'
RAW_DATA_DIR=$ROOT/data/raw
CLEAN_DATA_DIR=$ROOT/data
CKPTDIR=$ROOT/checkpoints


# Create directory structures
# mkdir -p $RAW_DATA_DIR
mkdir $ROOT/checkpoints
mkdir $ROOT/preds
mkdir -p $ROOT/scripts/resnext101v01/weights
mkdir -p $ROOT/scripts/resnext101v02/weights

# Download checkpoint
cd $CKPTDIR
wget https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth -O resnext101_32x8d_wsl_checkpoint.pth
cd $ROOT

# Unzip competition data 
cd $RAW_DATA_DIR
unzip -qq rsna-intracranial-hemorrhage-detection.zip
cd $ROOT

# Copy csv files to data directory
cp $RAW_DATA_DIR/*.csv* CLEAN_DATA_DIR/
unzip CLEAN_DATA_DIR/*.csv*

# Prepare images and metadata
python scripts/prepare_meta_dicom.py
python scripts/prepare_folds.py
