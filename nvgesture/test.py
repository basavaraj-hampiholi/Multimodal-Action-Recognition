import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary

from torchvideotransforms import video_transforms, volume_transforms
from models.i3dpt import I3D
from models.fusion_model import TransFusion
from dataset.nvdataset import test_dataset
from utils import accuracy, AverageMeter
from config_beide import *

import shutil
import time
import random
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print_freq=10
use_gpu = torch.cuda.is_available()

def load_weights(path, type):
    if type=='rgb':
      i3d = I3D(num_classes=400, modality='rgb')
      i3d.conv3d_0c_1x1 = nn.Linear(1024, num_classes)
      i3d.load_state_dict(torch.load(path))

    elif type=='depth':
      i3d = I3D(num_classes=400, modality='rgb')
      i3d.conv3d_0c_1x1 = nn.Linear(1024, num_classes)
      i3d.load_state_dict(torch.load(path))

    return i3d

def main():

    best_prec1=0
    i3d_rgb = load_weights('load_model/nv_rgb_best.pt', 'rgb')
    i3d_flow = load_weights('load_model/nv_depth_best.pt', 'depth')   

    model = TransFusion(copy.deepcopy(i3d_rgb), copy.deepcopy(i3d_flow), num_classes)   
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('save_model/nv_fusion_resnet.pt'))
    #summary(model, [(1,3,16,112,112),(1,3,16,112,112)])
    #exit(1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()                         
    cudnn.benchmark = True
                                                                                                                                                                    
    tst_dataset = test_dataset(nvgesture_root, lst_val_file, seg_len, False)                                                                                                                                                                    
    test_loader =  torch.utils.data.DataLoader(tst_dataset, batch_size=8,
                                               shuffle=True, num_workers=num_workers)

    start_time= time.time()
    epoch=0
    prec1 = validate(test_loader, model, criterion, epoch)
    print('Top Precision:',prec1)

    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)


def validate(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to validation mode
    model.eval()
    pred = []
    gt = []

    end = time.time()
    with torch.no_grad():
      for i, (input_rgb, input_depth, target) in enumerate(test_loader):
        
          if use_gpu:
             input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
             input_depth = torch.autograd.Variable(input_depth.float().cuda())
             target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))
          else:
             input_rgb = torch.autograd.Variable(input_rgb.float())
             input_depth = torch.autograd.Variable(input_depth.float())
             target = torch.autograd.Variable(target.long())

          # compute output
          output = model(input_rgb,input_depth)
          loss = criterion(output, target)

          out = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
          pred.extend(out)   
          labels = target.data.cpu().numpy()
          gt.extend(labels)       

          # measure accuracy and record loss
          prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
          losses.update(loss.data, input_rgb.size(0))
          top1.update(prec1, input_rgb.size(0))
          top5.update(prec5, input_rgb.size(0))

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()

          if i % print_freq == 0:
              print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     i, len(test_loader), batch_time=batch_time, loss=losses,
                     top1=top1, top5=top5))

      print(' Test:{0} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(epoch, top1=top1, top5=top5))

      results = open('cnn_valid.txt', 'a')
      results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
            .format(epoch, loss=losses, top1=top1, top5=top5))
      results.close()

      np.save('nv_gt.npy', gt)
      np.save('nv_pred.npy', pred)

      cf_matrix = confusion_matrix(gt, pred)
      df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis=1), index = [i+1 for i in range(0,25)],
                     columns = [i+1 for i in range(0,25)])
      plt.figure(figsize = (10,8))
      sns.heatmap(df_cm, annot=True, annot_kws={"size": 5}, linewidths=0.1, linecolor='gray')
      plt.savefig('iso_confusion.pdf')
      
      return top1.avg



if __name__ == '__main__':
   main()
