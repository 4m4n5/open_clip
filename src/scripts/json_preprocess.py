import argparse
import os
import pickle
import platform
from typing import Any, List


from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import pandas as pd
import json

# fmt: off
parser = argparse.ArgumentParser("Convert ALBEF jsons to csvs to train with CLIP.")
parser.add_argument(
    "-o",
    "--output-dir",
    default="csvs/",
    help="Path to to the folder which stores the file containing serialized dataset.",
)
# fmt: on


if __name__ == "__main__":

    json_files = ['data/coco_karpathy_train.json',
                  'data/vg_caption.json',
                  'data/cc12m.json',
                  'data/conceptual_caption_train.json',
                  'data/conceptual_caption_val.json',
                  'data/sbu_caption.json'
                  ]

    data = []
    for f in json_files:
        data += json.load(open(f, 'r'))

    _A = parser.parse_args()
    os.makedirs(os.path.dirname(_A.output_dir), exist_ok=True)

    d = {}
    i = 0

    for idx, batch in enumerate(tqdm(data)):
        image_id = idx
        filename = batch["image"]
        captions = batch["caption"]

        if type(captions) == list:
            for caption in captions:
                # add a dictionary entry to the final dictionary
                d[i] = {"img_path": filename, "caption": caption}
                # increment the counter
                i = i + 1
        else:
            caption = captions
            d[i] = {"img_path": filename, "caption": caption}
            i = i + 1
        if idx >= 1000:
            break

    data = pd.DataFrame.from_dict(d, "index")
    output_path = os.path.join(_A.output_dir, "json_data.csv")
    data.to_csv(output_path)
