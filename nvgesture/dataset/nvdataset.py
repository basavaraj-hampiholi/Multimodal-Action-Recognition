import os
import numpy as np
from torch.utils.data import Dataset
from torchvideotransforms import video_transforms, volume_transforms
import skimage.io as imageio
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

    def __init__(self, root_dir, lst_file_input, seg_len, train, transform=None):
        self.root = root_dir
        self.lst_file_input = lst_file_input
        self.files_list = load_split_nvgesture(self.lst_file_input)  

        self.transform = transform
        self.train = train
        self.seg_len = seg_len
        self.image_tmpl='frame{:05d}.jpg'

    def __len__(self):
        return len(self.files_list)

    def normalize(self, feats):
        mean = feats.mean()
        std = feats.std()
        normal_feats = (feats - mean)/std
        return normal_feats

    def read_video(self, path, idx_list):
        vid_path = path[2:] 
        full_path = os.path.join(self.root, vid_path)

        videos = []
        for i in idx_list:
            img_path = os.path.join(full_path, self.image_tmpl.format(i))
            frame = imageio.imread(img_path)
            videos.append(frame)

        return videos

    def make_dataset(self, rgb_path):
        vid_path = rgb_path[2:]
        full_path = os.path.join(self.root, vid_path)

        video_len = len(os.listdir(full_path))
        if self.train:
            if video_len>=self.seg_len:
                tot_len = abs(video_len-self.seg_len)
                idx_list = np.array(sorted(list(range(self.seg_len))))+random.randint(0, tot_len)
            else:
                tick = video_len/float(self.seg_len)
                idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(0, self.seg_len)]))
                return idx_list
        else:
            tick = video_len/float(self.seg_len)
            idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(0, self.seg_len)]))

        return idx_list
    
    
    def __getitem__(self, idx):
        
        rgb_path = self.files_list[idx]['color']
        depth_path = self.files_list[idx]['depth']
        label = self.files_list[idx]['label']
        
        rgb_list = self.make_dataset(depth_path)
        rgb_video = self.read_video(rgb_path, rgb_list)     
        depth_video = self.read_video(depth_path, rgb_list)
        
        if self.transform:
            rgb_video = self.transform(rgb_video)
            depth_video = self.transform(depth_video)

        return rgb_video, depth_video, label

def training_dataset(root_dir, lst_file_input, seg_len, train):


    nv_train_dataset = NVGestureDataset(root_dir, lst_file_input, seg_len, train, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), video_transforms.RandomCrop(224), volume_transforms.ClipToTensor()]))
                              

    return nv_train_dataset


def test_dataset(root_dir, lst_file_input, seg_len, train):

    nv_test_data = NVGestureDataset(root_dir, lst_file_input, seg_len, train, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), video_transforms.CenterCrop(224), volume_transforms.ClipToTensor()]))
                                                                                                          
    return nv_test_data
