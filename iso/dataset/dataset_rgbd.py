import os
import torch
import pandas as pd
import numpy as np
import cv2
import skimage.io as imageio
import skvideo.io as vio
import torchvision.io as tio
from skimage.util import crop
from torch.utils.data import Dataset, DataLoader
from skimage.transform import rescale, resize
from torchvideotransforms import video_transforms, volume_transforms
#from vidaug import augmentors as va
import torchvision.transforms as transforms
import random

class IsoGDDataset(Dataset):
    def __init__(self, root_dir, csv_path, seg_len, train, transform=None):       
        self.root_dir = root_dir
        self.model_data = pd.read_csv(csv_path)
        self.transform = transform
        self.seg_len = seg_len
        self.clip_min_sz = 8
        self.train=train
        self.image_tmpl="{:06d}.jpg"

    def __len__(self):
        return len(self.model_data)

    def normalize(self, feats):

        mean = torch.mean(feats)
        std = torch.std(feats)
        normal_feats = (feats - mean)/std

        return normal_feats

    def clip_video(self, video, path):
        vid_len = video.size(1)
        video_tensor = video
        if vid_len>self.seg_len:
           video_tensor = video_tensor[:,0:self.seg_len,:,:]
        if vid_len<=self.seg_len:
            rem_len = self.seg_len-vid_len
            for k in range(0,rem_len):
                video_tensor = torch.cat((video_tensor, video[:,vid_len-1,:,:].unsqueeze(1)),dim=1)

        return video_tensor

    def read_video(self, vid_path, frames):

        video = []
        vid_path = vid_path[:-4]
        folder = vid_path.split('/')[-1]
        #print(vid_path, folder+'_'+self.image_tmpl.format(1))
        #exit(1)
        for i in sorted(frames):
            img_path = os.path.join(vid_path, folder+'_'+self.image_tmpl.format(i+1))
            img = imageio.imread(img_path)
            video.append(img)
        video = np.array(video)

        return video

    def make_dataset(self, rgb_path):
        vid_path = os.path.join(self.root_dir, rgb_path[:-4])
        video_len = len(os.listdir(vid_path))-1

        if self.train:
            if video_len>=self.seg_len:
                tot_len = abs(video_len-self.seg_len)
                idx_list = np.array(sorted(list(range(self.seg_len))))+random.randint(0,tot_len)
            else:
                tick = video_len/float(self.seg_len)
                idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(self.seg_len)]))
                return idx_list
        else:
            tick = video_len/float(self.seg_len)
            idx_list = np.array(sorted([int(tick/2.0 + tick * x) for x in range(self.seg_len)]))

        return idx_list

    def __getitem__(self, idx):

        rgb_path = os.path.join(self.root_dir, self.model_data.loc[idx,'rgb'])
        depth_path = os.path.join(self.root_dir, self.model_data.loc[idx,'depth'])

        rgb_list = self.make_dataset(rgb_path)
        #depth_list = self.make_dataset(depth_path)
            
        rgb_vid = self.read_video(rgb_path, rgb_list)
        depth_vid = self.read_video(depth_path, rgb_list)

        label = self.model_data.loc[idx,'class']-1 

        if self.transform:
            rgb_vid = self.transform(rgb_vid)
            depth_vid = self.transform(depth_vid)

        #rgb_vid = self.clip_video(rgb_vid, rgb_path)
        #depth_vid = self.clip_video(depth_vid, rgb_path)

        return rgb_vid, depth_vid, label

def augment_dataset(drive_root, csv_train, seg_len):
    video_rotate_augment = [video_transforms.Resize(256),
                            video_transforms.RandomCrop(224),
                            video_transforms.RandomHorizontalFlip(), volume_transforms.ClipToTensor()]
                            
    orig_dataset = IsoGDDataset(drive_root, csv_train, seg_len, True,
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                              video_transforms.RandomCrop(224), volume_transforms.ClipToTensor()]))


    return orig_dataset

def test_dataset(drive_root, csv_test, seg_len):
    test_data = IsoGDDataset(drive_root, csv_test, seg_len, False,
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                                                       video_transforms.CenterCrop(224), 
                                                       volume_transforms.ClipToTensor()]))
    return test_data
