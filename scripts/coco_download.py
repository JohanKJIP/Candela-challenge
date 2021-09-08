"""
    Download boat images from COCO dataset.
"""

import fiftyone as fo

dataset = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["boat"],
    max_samples=5000,
)

# Visualize the dataset in the FiftyOne App
session = fo.launch_app(dataset)
session.wait()
