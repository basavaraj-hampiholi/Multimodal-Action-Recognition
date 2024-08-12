# NVGesture dataset
    NVGesture [1] is an in-car dynamic hand gesture recognition dataset captured from multiple viewpoints using multiple sensors. NVGesture contains gesture videos of three different modalities viz. RGB, depth, and infrared (IR). The videos are recorded at the rate of 30fps and with a resolution of 320×240 pixels. The dataset consists of 1,532 gesture videos with 25 different classes of hand gestures. The dataset is split into a training set with 1,050 samples and a test set with 482 samples. The gestures were captured from 20 different individuals. IR videos do not have the same viewpoint as RGB and depth videos. Therefore, we use only RGB and depth modalities for our experiments.

## Download
    Download this dataset from - https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d
    Unzip videos using the command:-    
    
            7z x nvGesture_v1.7z.001
            
    This creates a directory named Video_data

    
## Prepare Dataset: Extract Frames from Videos
    The videos in NVGesture dataset contain the action in a specific time frame. Using the following script, we extract the frames of both modalities between the time 
    frame. Specify the absolute path of the downloaded directory (one dir above Video_data)
    
            python readdata.py --datadir '/path/to/nvgesture/'

    In the end, we have 80 frames for each video sample. 
