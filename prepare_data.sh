# Parameters
# Clone repo https://github.com/darraghdog/rsna and set the location as ROOT directory
ROOT='/mnt/lsf/share/dhanley2/rsna'
RAW_DATA_DIR=$ROOT/data/raw
CLEAN_DATA_DIR=$ROOT/data

# Create directory structures
mkdir -p $RAW_DATA_DIR
mkdir $ROOT/checkpoints

# Download competition data
cd $RAW_DATA_DIR
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
unzip rsna-intracranial-hemorrhage-detection
cd $ROOT

# Prepare images and metadata
python rsna/eda/window_meta2csv.py
python rsna/eda/window_v1.py 
