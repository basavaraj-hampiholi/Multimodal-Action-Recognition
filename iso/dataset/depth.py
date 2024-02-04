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
        min_d = feats.min()
        max_d = feats.max()
        normal_feats = (feats - min_d)/(max_d- min_d)
        return normal_feats

    def read_video(self, vid_path):
        video = []
        vid_path = vid_path[:-4]
        for i,img_file in enumerate(sorted(os.listdir(vid_path))):
            if i<self.seg_len:
                img_path = os.path.join(vid_path,img_file)
                img = imageio.imread(img_path)
                video.append(img)
        video = np.array(video)
        return video


    def clip_video(self, video, path):
        vid_len = video.size(1)
        video_tensor = video
        #if vid_len>self.seg_len:
        #    video_tensor = video_tensor[:,0:self.seg_len,:,:]
        if vid_len<=self.seg_len:
            rem_len = self.seg_len-vid_len
            for k in range(0,rem_len):
                video_tensor = torch.cat((video_tensor, video[:,vid_len-1,:,:].unsqueeze(1)),dim=1)
        return video_tensor


    def __getitem__(self, idx):
        
        depth_path = os.path.join(self.root_dir, self.model_data.loc[idx,'depth'])

        depth_vid = self.read_video(depth_path)
        label = self.model_data.loc[idx,'class']-1   

        if self.transform:
            depth_vid = self.transform(depth_vid)

        depth_vid = self.clip_video(depth_vid, depth_path)

        return depth_vid, label

def augment_dataset(drive_root, csv_train, seg_len):

    orig_dataset = IsoGDDataset(drive_root, csv_train, seg_len, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                              video_transforms.RandomCrop(224), volume_transforms.ClipToTensor()]))

    return orig_dataset

def test_dataset(drive_root, csv_test, seg_len):
    test_data = IsoGDDataset(drive_root, csv_test, seg_len, 
                    transform=video_transforms.Compose([video_transforms.Resize(256), 
                                                       video_transforms.CenterCrop(224), 
                                                       volume_transforms.ClipToTensor()]))
    return test_data