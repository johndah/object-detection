# MRI brain tumor detection

Localizing and classifing brain tumours with bounding boxes on brain MRI data set.
Documentation and implementation in [mri-detection.ipynb](mri-detection.ipynb)

## Motivation
Model, data augmentation and loss is inspired by [Ultralytics Yolo5](https://docs.ultralytics.com/yolov5/), which also is evaluated for comparison.
By replicating the original model, as opposed to reprodicing it, the implementation and performance differ. 
The intention behind this repo however is to get an better understanding of the object detection task by implementing a solution from scratch (using PyTorch etc.), why more focus on explainable code and visualizations rather than obtaining the exact same performace.