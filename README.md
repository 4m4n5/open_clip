# OpenCLIP

### Install dependencies

```bash
conda env create
source activate open_clip

# Additional packages
pip install opencv-python
```

### Add directory to pythonpath:

```bash
cd open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

### Prepare dataset
#### Preprocess training data from json files

1. Download the json files from the link provided in the ALBEF repo - https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip

2. Go to line number ``29`` in ``src/scripts/json_preprocess.py`` and provide paths to the json files downloaded in step 1.

3. Run the following command. This will create a csv file `json_data.csv` and puts it in the folder named `csvs`.

```bash
python src/scripts/json_preprocess.py 
```

#### Preprocess training and validation for COCO

The following commands will create csv files `csvs/val_coco.csv` and `csvs/train_coco.csv` which we will use for training and validation.

```bash
python src/scripts/coco_preprocess.py --split train --data-root /path/to/coco/dataset/
```

```bash
python src/scripts/coco_preprocess.py --split val --data-root /path/to/coco/dataset/
```

### Start training:

```bash
python -u src/training/main.py \
    --train-data="/path/to/json_data.csv"  \
    --val-data="/path/to/val_coco.csv"  \
    --warmup 10000 \
    --batch-size=128 \
    --lr=5e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=4 \
    --model RN50
```

Logging will be done on wandb
