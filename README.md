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
### Pre-process COCO data

The following commands will create two csv files `train_coco.csv` and `val_coco.csv` and puts them in the new folder named `csvs`.

```bash
python src/scripts/coco_preprocess.py --split train --data-root /path/to/coco/dataset/ 
python src/scripts/coco_preprocess.py --split val --data-root /path/to/coco/dataset/
```

### Start training on COCO data:

```bash
python -u src/training/main.py --train-data="/path/to/train_coco.csv" --val-data="/path/to/val_coco.csv"
```

Logging will be done on wandb

### Pre-process JSON data (to train on ALBEF data)

1. Download the json files from the link provided in the ALBEF repo - https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip

2. Go to line number ``29`` in ``src/scripts/json_preprocess.py`` and provide paths to the json files downloaded in step 1.

3. Run the following command. This will create a csv file `json_data.csv` and puts it in the folder named `csvs`.

```bash
python src/scripts/json_preprocess.py 
```

### Start training on JSON data (to train on ALBEF data):

```bash
python -u src/training/main.py --train-data="/path/to/json_data.csv" --val-data="/path/to/val_coco.csv"
```

Logging will be done on wandb
