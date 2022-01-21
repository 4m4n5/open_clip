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
## Pre-process COCO data
The following commands will create two csv files `train_coco.csv` and `val_coco.csv` and puts them in the new folder named `csvs`.

```bash
python src/scripts/coco_preprocess.py --split train --data-root /path/to/coco/dataset/ 
python src/scripts/coco_preprocess.py --split val --data-root /path/to/coco/dataset/
```

### Start training:

```bash
python -u src/training/main.py --train-data="/path/to/train_coco.csv" --val-data="/path/to/val_coco.csv"
```

Logging will be done on wandb
