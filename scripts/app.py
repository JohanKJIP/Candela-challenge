"""
    Visualise a labelled dataset in the browser.
"""

import fiftyone as fo
import fiftyone.zoo as foz

# Load downloaded dataset
DATASET_DIR = "/home/johan/fiftyone/coco-2017/train/"
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=DATASET_DIR,
    labels_path="/home/johan/fiftyone/coco-2017/train/labels.json",
    include_id=True,
)

# Visualize the dataset in the FiftyOne App
session = fo.launch_app(dataset)
session.wait()
