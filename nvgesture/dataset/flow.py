from config import *
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import skvideo.io as vio
import skimage.io as imageio
from skimage.util import crop
from skimage.transform import rescale, resize
from torchvideotransforms import video_transforms, volume_transforms
import pandas as pd
from math import floor
import random
import warnings
warnings.filterwarnings("ignore")

def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst'):
    params_dictionary = dict()
    files_list = []
    with open(file_with_split,'rb') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.decode().split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            files_list.append(params_dictionary)
 
    return np.array(files_list)

class NVGestureDataset(Dataset):

    def __init__(self, root_dir, lst_file_input, clip_len, transform=None):
        self.root = root_dir
        self.lst_file_input = lst_file_input
        self.files_list = load_split_nvgesture(self.lst_file_input)  
        self.transform = transform
        self.image_tmpl='frame{:05d}.jpg'
        self.train = True
        self.seg_len = clip_len

    def __len__(self):
        return len(self.files_list)

    def normalize(self, feats):
        mean = feats.mean()
        std = feats.std()
        normal_feats = (feats - mean)/std
        return normal_feats

    def read_flow(self, vid, idx_list):
        flows = []
        for i in idx_list:
            frame = vid[i,:,:,:]
            flows.append(frame)
        return np.array(flows)

    def make_dataset(self, vid):
        video_len = len(vid)
        if self.train:
            if video_len>=self.seg_len:
                tot_len = abs(video_len-(self.seg_len))
                idx_list = np.array(sorted(list(range(self.seg_len))))+random.randint(0, tot_len)
            else:
                tick = video_len/float(self.seg_len)
                idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(self.seg_len)]))
                return idx_list
        else:
            tick = video_len/float(self.seg_len)
            idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(self.seg_len)]))

        return idx_list

    def __getitem__(self, idx):
        file_path = self.files_list[idx]['color'][:-8]
        label = self.files_list[idx]['label']
        vid = vio.vread(self.root+file_path[1:]+'sk_flow.mp4')

        
        idx_list = self.make_dataset(vid)
        flow_vid = self.read_flow(vid, idx_list) 

        if self.transform:
            flow_vid = self.transform(flow_vid)

        return flow_vid[0:2,:,:,:], label

def training_dataset(root_dir, lst_file_input, seg_len, train):
    # video_rotate_augment = [video_transforms.Resize(256),
    #                         video_transforms.RandomCrop(224),
    #                         video_transforms.RandomRotation(45),
    #                         volume_transforms.ClipToTensor()]

    orig_dataset = NVGestureDataset(root_dir, lst_file_input, seg_len, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), video_transforms.RandomCrop(224),volume_transforms.ClipToTensor()]))
                              

    #flip_dataset = IsoGDDataset(drive_root, csv_train, seg_len, train, 
    #               transform=video_transforms.Compose(video_rotate_augment))
    
    #merged_dataset = torch.utils.data.ConcatDataset([orig_dataset, flip_dataset])

    return orig_dataset


def test_dataset(root_dir, lst_file_input, seg_len, train):
    test_data = NVGestureDataset(root_dir, lst_file_input, seg_len,
                    transform=video_transforms.Compose([video_transforms.Resize(256), video_transforms.CenterCrop(224),volume_transforms.ClipToTensor()]))
                                                                                                          
    return test_data
