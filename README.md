# Detection Summary Engine
## Vivek's Computer Vision based detection engine

The detection engine uses a pre-trained object detection model, yolov10, processes every frame of a short video, outputs a JSON file with
* Class labels
* Bounding box co-ordinates
* Confidence scores

counts total objects per class, identifies the frame with maximum class diversity, visualizes object frequency with a bar chart, and finally saves annonated frames and outputs a compiled output video.

## Pre-requisites:- Rename the short video to 'input_video.mp4' in the current directory, and create two folders: 'output_video' and 'output_images' in the directory you run your code from

raw file - Detection_Summary_Engine.py

download pdf to view ipynb output
