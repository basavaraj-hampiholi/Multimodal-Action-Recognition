# IsoGD dataset

Download IsoGD dataset from - http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html

Isolated gestures (IsoGD) [2]  is a large-scale multi-modal gesture recognition dataset derived from the Chalearn gesture dataset (CGD 2011). IsoGD contains gesture videos of two different modalities viz. RGB and depth with a resolution of 320×240 pixels and a frame rate of 10 fps. The dataset is user independent, which means the participants in the training set are not repeated in the test set. In total, the dataset includes 47,933 RGB-D gesture videos with 249 gesture labels. The gestures were performed by 21 different individuals. The dataset is split into a training set with 35,878 samples from 17 participants, a validation set with 5,784 samples from 2 participants, and a test set with 6,271 samples from 2 participants.


### Training

- First phase of training:
   The code consists of three directories nvgesture, iso, and ipn that correspond to each of the datasets.
   1. Download the i3d weights from https://github.com/piergiaj/pytorch-i3d/tree/master/models and place them in the load_model directory. 
   2. Load the i3d weights and finetune individual models from scratch by executing train_i3d_single.py on each modality. Save the finetuned weights.
   3. After training individual models, use their weights to initialize two-stream network and train it via <b><i>train_i3d_two_stream.py</i></b>.
      
- Second phase of training:  
   1. If you want to skip the first phase, there are fine-tuned weights available for each modality (rbg_best and depth_best).
   2. Download those weights into nvgesture/load_model directory from - 
      https://drive.google.com/file/d/16okuJxGgzSqbbO4DE2GR4Nyr10ag_auF/view?usp=drive_link.
   3. Load these weights and initialize two-stream fusion network. 
   4. Execute <b><i>train_i3d_two_stream.py</i></b> to train the fusion model.
      This should start the training as per the configurations specified in <b><i>config_beide.py</i></b>.

### Evaluation

After the second phase of training, the weights are stored in save_model directory.
Load these weights and execute <b><i>test.py</i></b> file to record the predictions on test data.
