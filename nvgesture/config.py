### general settings ###
nvgesture_root = '/gdisk/ni/home/hampiholi/vision/projects/datasets/nvgesture'
lst_train_file = './meta/nvgesture_train_correct.lst'
lst_val_file = './meta/nvgesture_test_correct.lst'

#path to the checkpoint
ckpt_path = './save_model/nv_rgb_checkpoint.pt'
#default path to the best model
path2best_model = './save_model/nv_depth_squeeze_best.pt'


#training parameters.
lr = 0.005
lr_decay = 0.1
epochs = 30
weight_decay = 1e-4
train_batch_size = 8
test_batch_size = 16
num_workers = 8
num_classes = 25
clip_st = 8
clip_end = 72
