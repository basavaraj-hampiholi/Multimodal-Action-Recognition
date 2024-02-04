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
import torchvision.transforms as transforms
import random

class IsoGDDataset(Dataset):
    def __init__(self, root_dir, csv_path, seg_len, train, transform=None):       
        self.root_dir = root_dir
        self.model_data = pd.read_csv(csv_path)
        self.transform = transform
        self.seg_len = seg_len
        self.clip_min_sz = 8
        self.image_tmpl='frame{:06d}.jpg'
        self.train = train

    def __len__(self):
        return len(self.model_data)

    def normalize(self, feats):
        mean = feats.mean()
        std = feats.std()
        normal_feats = (feats - mean)/std
        return normal_feats

    def read_flows(self, flow_path_u, flow_path_v, frm_list):
        flow_video = []
        for i, image in enumerate(sorted(os.listdir(flow_path_u))):
            if i<self.seg_len:                
                u_img_pth = os.path.join(flow_path_u, image)
                v_img_pth = os.path.join(flow_path_v, image)
                u_img = np.expand_dims(imageio.imread(u_img_pth),axis=2)
                v_img = np.expand_dims(imageio.imread(v_img_pth),axis=2)
                uv_img = np.concatenate((u_img, v_img),axis=2)
                flow_img = np.concatenate((uv_img, v_img),axis=2)
                flow_video.append(flow_img)
        flow_video = np.array(flow_video)

        return flow_video

    def read_video(self, vid_path, frm_list):
        video = []
        #vid_path = vid_path[:-4]
        for i, image in enumerate(sorted(os.listdir(vid_path))):
            if i<self.seg_len:                
                img_path = os.path.join(vid_path, image)
                img = imageio.imread(img_path)
                video.append(img)
        video = np.array(video)

        return video

    def clip_video(self, video):
        vid_len = video.size(1)
        video_tensor = video
        if vid_len<self.seg_len:
            rem_len = self.seg_len-vid_len
            for k in range(0,rem_len):
                video_tensor = torch.cat((video_tensor, video[:,vid_len-1,:,:].unsqueeze(1)),dim=1)
        norm_tensor = self.normalize(video_tensor)
        return norm_tensor

    def make_dataset(self, rgb_path):

        video_len = len(os.listdir(rgb_path))-1
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

        rgb_path = os.path.join(self.root_dir+'/rgb', self.model_data.loc[idx,'videos'])
        flow_path_u = os.path.join(self.root_dir+'/flow/u', self.model_data.loc[idx,'videos'])
        flow_path_v = os.path.join(self.root_dir+'/flow/v', self.model_data.loc[idx,'videos'])
        
        idx_list = 0 #self.make_dataset(rgb_path)
        rgb_vid = self.read_video(rgb_path, idx_list)
        flow_vid = self.read_flows(flow_path_u, flow_path_v, idx_list)

        label = self.model_data.loc[idx,'labels']  

        if self.transform:
            rgb_vid = self.transform(rgb_vid)
            flow_vid = self.transform(flow_vid)
            
        rgb_vid = self.clip_video(rgb_vid)
        flow_vid = self.clip_video(flow_vid)

        return rgb_vid, flow_vid[0:2,:,:,:], label

def training_dataset(drive_root, csv_train, seg_len, train):
    video_rotate_augment = [video_transforms.Resize(256),
                            video_transforms.RandomCrop(224),
                            video_transforms.RandomRotation(45),
                            volume_transforms.ClipToTensor()]

    orig_dataset = IsoGDDataset(drive_root, csv_train, seg_len, train,
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                              video_transforms.RandomCrop(224), volume_transforms.ClipToTensor()]))

    #flip_dataset = IsoGDDataset(drive_root, csv_train, seg_len, train, 
    #               transform=video_transforms.Compose(video_rotate_augment))
    
    #merged_dataset = torch.utils.data.ConcatDataset([orig_dataset, flip_dataset])

    return orig_dataset


def test_dataset(drive_root, csv_test, seg_len, train):
    test_data = IsoGDDataset(drive_root, csv_test, seg_len, train, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                                                       video_transforms.CenterCrop(224), 
                                                       volume_transforms.ClipToTensor()]))
    return test_data
