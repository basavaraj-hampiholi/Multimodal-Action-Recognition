# Convolutional Transformer Fusion Blocks for Multi-Modal Gesture Recognition

## Introduction
Gesture recognition defines an important information channel in human-computer interaction. Intuitively, combining inputs from multiple modalities improves the recognition rate. In this work, we explore multi-modal video-based gesture recognition tasks by fusing spatio-temporal representation of relevant distinguishing features from different modalities. We present a self-attention based transformer fusion architecture to distill the knowledge from different modalities in two-stream convolutional neural networks (CNNs). For this, we introduce convolutions into the self-attention function and design the Convolutional Transformer Fusion Blocks (CTFB) for multi-modal data fusion. These fusion blocks can be easily added at different abstraction levels of the feature hierarchy in existing two-stream CNNs. In addition, the information exchange between two-stream CNNs along the feature hierarchy has so far been barely explored. We propose and evaluate different architectures for multi-level fusion pathways using CTFB to gain insights into the information flow between both streams.

## How to run the experiments
First, clone the repository using:-

git clone https://github.com/basavaraj-hampiholi/Multimodal-Action-Recognition.git

Instructions are provided in folders to run the experiments for respective datasets.

## References
1. P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, and J. Kautz, "Online detection and classification of dynamic hand gestures with recurrent 3D convolutional neural networks", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4207-4215, Jun. 2016.
2. J. Wan, S. Z. Li, Y. Zhao, S. Zhou, I. Guyon, and S. Escalera, "ChaLearn looking at people RGB-D isolated and continuous datasets for gesture recognition", Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), pp. 761-769, Jun. 2016.
3. G. Benitez-Garcia, J. Olivares-Mercado, G. Sanchez-Perez and K. Yanai, "IPN hand: A video dataset and benchmark for real-time continuous hand gesture recognition", Proc. 25th Int. Conf. Pattern Recognit. (ICPR), pp. 4340-4347, Jan. 2021.
   
## Acknowledgements
1. We thank piergiaj (https://github.com/piergiaj/pytorch-i3d) for the trained weights of i3d (open source). We use them to initialize individual streams.
2. We use a piece of code for an efficient attention module from - https://github.com/cmsflash/efficient-attention. We modify it to process video data directly.
3. We adapt the benchmark code of the IPN Hand dataset to our experiments (https://github.com/GibranBenitez/IPN-hand) .
