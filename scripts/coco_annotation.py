"""
    Convert COCO annotations into x1, y1, x2, y2, cls_id
"""

import json
from collections import defaultdict
import os
import shutil
from tqdm import tqdm


# Options
json_file_path = '/home/johan/fiftyone/coco-2017/validation/labels.json'
images_dir_path = '/home/johan/fiftyone/coco-2017/validation/data'
output_path = 'eval.txt'

# Load labels file
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

# Generate new labels
images = data['images']
annotations = data['annotations']

for ant in tqdm(annotations):
    img_id = ant['image_id']
    name = os.path.join(images_dir_path, '{:012d}.jpg'.format(img_id))
    cat = ant['category_id']

    # Only keep boat labels
    if cat == 9:
        name_box_id[name].append([ant['bbox'], cat-9])


def to_training():
    """ Convert COCO to training label format. """
    with open(output_path, 'w') as f:
        for key in tqdm(name_box_id.keys()):
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])

                box_info = " %d,%d,%d,%d,%d" % (
                    x_min, y_min, x_max, y_max, int(info[1]))
                f.write(box_info)
            f.write('\n')


def to_eval():
    """ Convert COCO to eval label format. """
    folder = 'ground-truth'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    for key in tqdm(name_box_id.keys()):
        print(key)
        with open(f'{folder}/{key.split("/")[-1].split(".")[0]}.txt', 'w') as f:
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])
                box = f'boat {x_min} {y_min} {x_max} {y_max}\n'
                f.write(box)


# Save new labels to file
to_eval()
        