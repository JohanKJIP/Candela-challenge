# Candela speedboat challenge

## Task
Using any machine learning platform and programming language of your choice:

    i) Write an algorithm to detect and count the different boats in realtime.
    ii) Save the video with a boundingbox for all the boats.

## Design choices
The challenge is time-limited which affected a few of the design choices. In particular, I chose to work with frameworks I was already familiar with to reduce time spent on environment setup. Additionally, I chose the Python programming language because it allows for quick prototyping. 

I chose to use YOLOv4 for the object detection task because it performs well for real-time tasks. In terms of mAP, YOLOv4-608 is #12 on the top list for [Real-Time Object detection on COCO](https://paperswithcode.com/sota/real-time-object-detection-on-coco). I am also already familiar with the implementation from my time working with KTH Formula Student (KTHFS) where we used YOLOv4-416. In particular, I based my code on [this](https://github.com/Tianxiaomo/pytorch-YOLOv4) PyTorch library. For weights I browsed the [DarkNet YOLOv4 model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo) and downloaded config- and pretrained weight files for the COCO dataset.

Because it is a real-time task I chose to utilise [TensorRT](https://developer.nvidia.com/tensorrt) to speed up the inference time of the neural network. In my experience, this speeds up the inference time by 2-4x for YOLO. With TensorRT using FP16 precision, a frame takes around 13.7ms to process on an RTX 3070 which converts to ~74FPS (including tracking). 

The output of the YOLO network can be a bit noisy and it has no data association between frames. From my time at KTHFS I was familiar with the idea of tracking bounding boxes with data association and Kalman filters. I did not have time to implement it for KTHFS but wanted to give it a try with this challenge. So, I started implementing data association between frames from scratch by comparing Intersection Over Union (IOU) between the bounding boxes from time t-1 and t. My next plan was to research Kalman Filters and how they could be used with my data association. However, as I was researching it, I stumbled across [SORT](https://github.com/abewley/sort) on Github. SORT is a real-time tracking algorithm that implements the data association I had worked on but it also takes care of the Kalman filters for each tracker. Due to the time constraints, I chose to abandon my implementation and integrate SORT instead. 


## Challenges
1. Self detections: One challenge with the video is the issue of the boat carrying the camera being detected and "counted" as a boat.

    **Solution**: At first, the issue was solved by removing bounding boxes with a very large pixel area. This removes the self detection because "our" boat covers a large portion of the screen. This approach limits the programs ability to detect boats that are large in frame, e.g. very close to us or just very large. For other videos it severly limits the programs ability to detect large boats. 

    The second approach utilises an exclusion zone where "our" boat is located in the frame (shown as a white transparent square in the bounding box video). Any bounding box with a center inside the exclusion zone is removed. With the flag `--no-filter` the exclusion zone can be removed.

2. CUDA out of memory

    todo

## Future work

1. **Improve tracking**: The tracking of bounding boxes works quite well with SORT. However, there is a new version called [Deep SORT](https://github.com/nwojke/deep_sort) that could perform better. 

2. **More training data**: The COCO dataset has approximately 3000 images of boats. I think it could be complemented with a dataset containing images from the camera on the boat. 

3. **Training with GPU with more VRAM**: I only have access to a GPU with 8GB of VRAM. Alexey recommends 16GB to train a full YOLOv4 network with batch size=64 and subdivisions=16. You can reduce the batch size or input size but that will most likely result in lower performance. 

4. **Multi-class for different types of boats**: It could potentially be useful to know what type of boat we have detected. A sail boat will behave differently from a cruise ship which needs to be considered when moving autonomously. 


## Setup
The following are instructions for running the code. 

### Prerequisites
- Have CUDA capable GPU
- Install CUDA, cudaNN and TensorRT (has been tested with version 7.2.3)
- Python 2 for running inference (might work with Python 3 but I am using v2 because that is what I had set up. For training with the yolov4 module you have to use Python 3)
- Install inference requirements file: `python -m pip install -r requirements.txt`
- Install requirements for the submodules `sort` and `yolov4`
- Add `__init__.py` inside sort the directory

### Downloading weights and building TensorRT engine
- Download cfg file and weights file for YOLOv4-608 from the [YOLOv4 model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo).
- Place the config file in the `yolov4/cfg` directory.
- Place the weight file in the `yolov4` directory. 
- Convert model to ONNX format. Run `python3 demo_darknet2onnx <path to cfg> <path to weight file> <path to test image> <batch size (1 in this case)>`
- Move the converted ONNX model back to the root of the repository.
- Build the TensorRT engine from the ONNX file. `python build_trt_engine.py <path to onnx> yolov4.trt <mode (fp16 or fp32)>`

### Running

To run the detector you type `python video_detect.py <flags>`. There are two flags: the path video to detect boats in and if self detections should be filtered or not.

For example: `python video_detect.py --video videos/example.mkv --no-filter` will run detections on example.mkv and will not remove self detections. 

