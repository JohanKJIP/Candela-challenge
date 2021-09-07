# Candela speedboat challenge

## Task
Using any machine learning platform and programming language of your choice:

    i) Write an algorithm to detect and count the different boats in realtime.
    ii) Save the video with a boundingbox for all the boats.

## Design choices
The challenge is time limited which affected a few of my design choices. In particular, I chose to work with frameworks I am already familiar with to reduce time spent on environment setup. Additionally, I chose the Python programming language because it allows for quick prototyping. 

I chose to use YOLOv4 for the object detection task because it performs well for real-time tasks. In terms of mAP, YOLOv4-608 is #12 on the toplist for [Real-Time Object detection on COCO](https://paperswithcode.com/sota/real-time-object-detection-on-coco). I am also already familiar with the implementation from my time working with KTH Formula Student (KTHFS) where we used YOLOv4-416. In particular, I based my code on [this](https://github.com/Tianxiaomo/pytorch-YOLOv4) PyTorch library. 

Because it is a real-time task I chose to utilise [TensorRT](https://developer.nvidia.com/tensorrt) to speed up the inference time of the neural network. In my experience, this speeds up the inference time by 2-4x for YOLO. With TensorRT using FP16 precision, a frame takes around 13.7ms to process on an RTX 3070 which converts to ~74FPS (including tracking). 

The output of the YOLO network can be a bit noisy and it has no data association between frames. From my time at KTHFS I was familiar with the idea tracking bounding boxes with data association and Kalman filters. I did not have time to implement it for KTHFS but wanted to give it a try with this challenge. So, I started implementing data association between frames from scratch by comparing Intersection Over Union (IOU) between the bounding boxes from time t-1 and t. My next plan was to research Kalman Filters and how they could be used with my data association. However, as I was researching it, I stumbled across [SORT](https://github.com/abewley/sort) on Github. SORT is a realtime tracking algorithm that implements the data association I had worked on but it also takes care of the Kalman filters for each tracker. Due to the time constraints I chose to abandon my own implementation and integrate SORT instead. 


## Challenges
1. Self detections: One challenge with the video is the issue of the boat carrying the camera being detected and "counted" as a boat.

    **Solution**: At first, the issue was solved by removing bounding boxes with a very large pixel area. This removes the self detection because "our" boat covers a large portion of the screen. This approach limits the programs ability to detect boats that are large in frame, e.g. very close to us or just very large. For other videos it severly limits the programs ability to detect large boats. 

    The second approach utilises an exclusion zone where "our" boat is located in the frame (shown as a white transparent square in the bounding box video). Any bounding box with a center inside the exclusion zone is removed. With the flag `--no-filter` the exclusion zone can be removed.

2. Background can affect boat detection

    todo

3. CUDA out of memory

    todo

## Future work

Deep sort: https://github.com/nwojke/deep_sort.

## Setup