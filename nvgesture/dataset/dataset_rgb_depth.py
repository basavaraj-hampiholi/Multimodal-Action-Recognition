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
from vidaug import augmentors as va
import torchvision.transforms as transforms

class IsoGDDataset(Dataset):
    def __init__(self, root_dir, csv_path, seg_len, transform=None):       
        self.root_dir = root_dir
        self.model_data = pd.read_csv(csv_path)
        self.transform = transform
        self.seg_len = seg_len
        self.clip_min_sz = 8

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
        elif vid_len<=self.seg_len:
            rem_len = self.seg_len-vid_len
            for k in range(0,rem_len):
                video_tensor = torch.cat((video_tensor, video[:,vid_len-1,:,:].unsqueeze(1)),dim=1)
        #norm_tensor = self.normalize(video_tensor)
        return video_tensor


    def __getitem__(self, idx):

        rgb_path = os.path.join(self.root_dir, self.model_data.loc[idx,'rgb'])
        depth_path = os.path.join(self.root_dir, self.model_data.loc[idx,'depth'])
            
        rgb_vid = vio.vread(rgb_path)
        depth_vid = vio.vread(depth_path)
        label = self.model_data.loc[idx,'class']-1 

        if self.transform:
            rgb_vid = self.transform(rgb_vid)
            depth_vid = self.transform(depth_vid)
        rgb_vid = self.clip_video(rgb_vid, rgb_path)
        depth_vid = self.clip_video(depth_vid, rgb_path)

        return rgb_vid, depth_vid, label

def augment_dataset(drive_root, csv_train, seg_len):
    video_rotate_augment = [video_transforms.Resize((128,171)),
                            video_transforms.RandomCrop(112),
                            va.Pepper(), va.Salt(), volume_transforms.ClipToTensor()]
                            
    orig_dataset = IsoGDDataset(drive_root, csv_train, seg_len, 
                    transform=video_transforms.Compose([video_transforms.Resize((128,171)), 
                              video_transforms.RandomCrop(112), volume_transforms.ClipToTensor()]))

    flip_dataset = IsoGDDataset(drive_root, csv_train, seg_len, 
                       transform=video_transforms.Compose(video_rotate_augment))
    
    #merged_dataset = torch.utils.data.ConcatDataset([orig_dataset, flip_dataset])

    return orig_dataset

def test_dataset(drive_root, csv_test, seg_len):
    test_data = IsoGDDataset(drive_root, csv_test, seg_len, 
                    transform=video_transforms.Compose([video_transforms.Resize(128), 
                                                       video_transforms.CenterCrop(112), 
                                                       volume_transforms.ClipToTensor()]))
    return test_data
