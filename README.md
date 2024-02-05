# Multimodal Action/Gesture Recognition in Videos

## Introduction
Gesture recognition defines an important information channel in human-computer interaction. Intuitively, combining inputs from multiple modalities improves the recognition rate. In this work, we explore multi-modal video-based gesture recognition tasks by fusing spatio-temporal representation of relevant distinguishing features from different modalities. We present a self-attention based transformer fusion architecture to distill the knowledge from different modalities in two-stream convolutional neural networks (CNNs). For this, we introduce convolutions into the self-attention function and design the Convolutional Transformer Fusion Blocks (CTFB) for multi-modal data fusion. These fusion blocks can be easily added at different abstraction levels of the feature hierarchy in existing two-stream CNNs. In addition, the information exchange between two-stream CNNs along the feature hierarchy has so far been barely explored. We propose and evaluate different architectures for multi-level fusion pathways using CTFB to gain insights into the information flow between both streams.

## Datasets

1. Download this dataset from - https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d
   
NVGesture [1] is an in-car dynamic hand gesture recognition dataset captured from multiple viewpoints using multiple sensors. NVGesture contains gesture videos of three different modalities viz. RGB, depth, and infrared (IR). The videos are recorded at the rate of 30fps and with a resolution of 320×240 pixels. The dataset consists of 1,532 gesture videos with 25 different classes of hand gestures. The dataset is split into a training set with 1,050 samples and a test set with 482 samples. The gestures were captured from 20 different individuals. IR videos do not have the same viewpoint as RGB and depth videos. Therefore, we use only RGB and depth modalities for our experiments.

2. Download IsoGD dataset from - http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html

Isolated gestures (IsoGD) [2]  is a large-scale multi-modal gesture recognition dataset derived from the Chalearn gesture dataset (CGD 2011). IsoGD contains gesture videos of two different modalities viz. RGB and depth with a resolution of 320×240 pixels and a frame rate of 10 fps. The dataset is user independent, which means the participants in the training set are not repeated in the test set. In total, the dataset includes 47,933 RGB-D gesture videos with 249 gesture labels. The gestures were performed by 21 different individuals. The dataset is split into a training set with 35,878 samples from 17 participants, a validation set with 5,784 samples from 2 participants, and a test set with 6,271 samples from 2 participants.

3. Download the IPN hand dataset from - https://gibranbenitez.github.io/IPN_Hand/

The IPN hand dataset [3] focuses on gestures that are relevant to interaction with touchless screens. It contains RGB videos with a resolution of 640×480 pixels recorded at 30fps using PC or laptop cameras. The videos are captured from 28 different scenes with 50 participants. These scenes include cluttered backgrounds and varying illumination. In total, there are 4,218 gesture instances with 13 different gesture classes. The dataset is slightly imbalanced with a majority of the samples belonging to only 2 classes. It is intended for both isolated and continuous gesture recognition tasks. We focus only on isolated gesture recognition. The dataset is randomly split into training (74%) and testing (26%) sets. For isolated gestures, the training set contains 3,117 gesture instances from 37 subjects and the test set contains 1,101 gesture instances from 13 subjects.

## How to run the code

First, clone the repository using - git clone -r https://github.com/basavaraj-hampiholi/Multimodal-Action-Recognition.git

The code consists of three directories nvgesture, iso, and ipn that correspond to each of the datasets.
Also, download the pre-trained weights into nvgesture/load_model directory from - https://drive.google.com/file/d/16okuJxGgzSqbbO4DE2GR4Nyr10ag_auF/view?usp=drive_link
Execute <b><i>train_i3d_two_stream.py</i></b> to train the fusion model. This should start the training as per the configurations specified in <b><i>config_beide.py</i></b>.
To run tests, execute <b><i>test.py</i></b> file.

To train individual models from scratch, execute train_i3d_single.py. Prior to this, download the i3d weights from https://github.com/piergiaj/pytorch-i3d/tree/master/models and place them in load_model directory. 
After training individual models, use their weights to initialize two-stream network and train it via <b><i>train_i3d_two_stream.py</i></b>. 

## References
1. P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, and J. Kautz, "Online detection and classification of dynamic hand gestures with recurrent 3D convolutional neural networks", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4207-4215, Jun. 2016.
2. J. Wan, S. Z. Li, Y. Zhao, S. Zhou, I. Guyon, and S. Escalera, "ChaLearn looking at people RGB-D isolated and continuous datasets for gesture recognition", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), pp. 761-769, Jun. 2016.
3. G. Benitez-Garcia, J. Olivares-Mercado, G. Sanchez-Perez and K. Yanai, "IPN hand: A video dataset and benchmark for real-time continuous hand gesture recognition", Proc. 25th Int. Conf. Pattern Recognit. (ICPR), pp. 4340-4347, Jan. 2021.
   
## Acknowledgements
1. We thank piergiaj (https://github.com/piergiaj/pytorch-i3d) for open sourcing the trained weights of i3d. We use them to initialize individual streams.
2. We also use a piece of code for an efficient attention module from - https://github.com/cmsflash/efficient-attention. We thank the authors for this. 
