### general settings ###
isogd_root = '/gdisk/ni/home/hampiholi/vision/projects/datasets/IsoGD1/frames'

csv_train = 'train.csv'
csv_val = 'valid.csv'
csv_test = 'test.csv'

#path to the saved model
ckpt_path = './save_model/isogd_rgb_checkpoint.pt'

#default path to the best model
path2best_model = './save_model/isogd_depth_224_best.pt'


#training parameters.
lr = 0.05
lr_decay = 0.1
epochs = 15
weight_decay = 1e-4
seg_len = 64
train_batch_size = 16
test_batch_size = 16
num_workers = 8
num_classes = 249

'''Train
#total samples = 6642
#after cleaning = 6295

val
total samples =1430
after cleaning = 1354

test
total samples=2222
after cleaning=2074 ''' 