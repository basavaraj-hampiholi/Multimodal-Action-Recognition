# NVGesture dataset

NVGesture [1] is an in-car dynamic hand gesture recognition dataset captured from multiple viewpoints using multiple sensors. NVGesture contains gesture videos of three different modalities viz. RGB, depth, and infrared (IR). The videos are recorded at the rate of 30fps and with a resolution of 320×240 pixels. The dataset consists of 1,532 gesture videos with 25 different classes of hand gestures. The dataset is split into a training set with 1,050 samples and a test set with 482 samples. The gestures were captured from 20 different individuals. IR videos do not have the same viewpoint as RGB and depth videos. Therefore, we use only RGB and depth modalities for our experiments.

## Download

Download this dataset from - https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d

Unzip videos using the command:-   

            7z x nvGesture_v1.7z.001

## Prepare Dataset: Extract Frames from Videos

The videos in the NVGesture dataset contain the action in a specific time frame. We extract the frames of both the modalities between the time frame using the following script:- 
            
            python readdata.py --datadir '/path/to/data/directory'

In the end, it creates directories with 80 frames for each of the video samples. 

## Test two-stream fusion network 

Download the checkpoints (mainly nv_fusion_best.pt) from https://drive.google.com/file/d/16okuJxGgzSqbbO4DE2GR4Nyr10ag_auF/view?usp=drive_link
and place them in the load_checkpoint directory. 
    
Execute the following command to evaluate the model:-

        CUDA_VISIBLE_DEVICES=0 python3 test_two_stream.py --datadir '/path/to/data/directory' --load_checkpoint './load_checkpoint/nv_fusion_best.pt'

## Train two-stream fusion network 

Download the checkpoints (mainly rgb_best.pt and depth_best.pt) from https://drive.google.com/file/d/16okuJxGgzSqbbO4DE2GR4Nyr10ag_auF/view?usp=drive_link
and place them in the load_checkpoint directory. 
    
Execute the following command to train the model using a single GPU:-

        CUDA_VISIBLE_DEVICES=0 python3 train_two_stream.py --datadir '/path/to/data/directory' 
    
Training using Multiple GPUs - execute the following command:- 

        CUDA_VISIBLE_DEVICES=0,1 python3 train_two_stream.py --datadir '/path/to/data/directory' --use_dataparallel True

## References

1. P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, and J. Kautz, "Online detection and classification of dynamic hand gestures with recurrent 3D convolutional neural networks", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4207-4215, Jun. 2016.

## Acknowledgements

1. We thank piergiaj (https://github.com/piergiaj/pytorch-i3d) for the trained weights of i3d (open source). We use them to initialize individual streams.

2. We also use a piece of code for an efficient attention module from - https://github.com/cmsflash/efficient-attention.
