import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary
from sklearn.utils.class_weight import compute_class_weight

from models.i3d import InceptionI3d
from models.fusion_central import TransFusion
from dataset.dataset_rgbd import IsoGDDataset, augment_dataset, test_dataset
from utils import accuracy, AverageMeter
from config_beide import *

import shutil
import time
import random
import numpy as np
import pandas as pd
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def save_checkpoint(state_dict, is_best, filename='./save_model/fusion_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, path2best_model)


print_freq=10
use_gpu = torch.cuda.is_available()

def loss_weights(train_csv, val_csv, test_csv):
    trn_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    tst_df = pd.read_csv(test_csv)
    full_df=pd.concat([trn_df, val_df, tst_df])
    class_list=list(full_df['class'])
    class_weights=compute_class_weight('balanced',np.unique(class_list),class_list)

    return torch.from_numpy(class_weights)

def load_weights(path, type):
     if type == 'rgb':
       i3d = InceptionI3d(400, in_channels=3)
       #i3d.load_state_dict(torch.load(path))
       i3d.replace_logits(num_classes)
       model = i3d.cuda()
       model.load_state_dict(torch.load('load_model/isogd_rgb_224_best.pt'))
     elif type=='flow':
       i3d = InceptionI3d(400, in_channels=3)
       #i3d.load_state_dict(torch.load(path))
       i3d.replace_logits(num_classes)
       model = i3d.cuda()
       model.load_state_dict(torch.load('load_model/isogd_depth_224_best.pt'))
     return model

def main():

    best_prec1=0
    i3d_rgb = load_weights('load_model/rgb_imagenet.pt','rgb')
    i3d_depth = load_weights('load_model/rgb_imagenet.pt', 'flow')      
     
    model = TransFusion(copy.deepcopy(i3d_rgb), copy.deepcopy(i3d_depth), num_classes)   
    model = nn.DataParallel(model).cuda()
    #model.load_state_dict(torch.load('load_model/isogd_fusion_best.pt'))
    #summary(model, [(3,64,224,224),(3,64,224,224)])
    #exit(1)

    train_dataset = augment_dataset(isogd_root, csv_train, seg_len)                                                                                                                                                                    
    valid_dataset = test_dataset(isogd_root, csv_val, seg_len)
                                                                                                                                                                          
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                               shuffle=True, num_workers=num_workers)

    val_loader =  torch.utils.data.DataLoader(valid_dataset, batch_size=test_batch_size,
                                               shuffle=True, num_workers=num_workers)

    # define loss function (criterion) and optimizer
    #class_weights=loss_weights(csv_train, csv_val, csv_test)
    #criterion = nn.CrossEntropyLoss(weight=class_weights.float().cuda(),reduction='mean').cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay, momentum=0.9, nesterov=True) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])                        
    cudnn.benchmark = True


    start_time= time.time()

    for epoch in range(0, epochs):

        # training on train dataset
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        print('Top Precision:',prec1)

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model.state_dict(), is_best, ckpt_path)

    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to train mode
    model.train()

    end = time.time()
    for i, (input_rgb, input_flow, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if use_gpu:
           input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
           input_flow = torch.autograd.Variable(input_flow.float().cuda())
           target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))

        else:
           input_rgb = torch.autograd.Variable(input_rgb.float())
           input_flow = torch.autograd.Variable(input_flow.float())
           target = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_rgb,input_flow)
        loss = criterion(output, target)

        #measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data, input_rgb.size(0))
        top1.update(prec1, input_rgb.size(0))
        top5.update(prec5, input_rgb.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    results = open('cnn_train.txt', 'a')
    results.write('Epoch: [{0}][{1}/{2}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
           epoch, i, len(train_loader), loss=losses,
           top1=top1, top5=top5))
    results.close()

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set to validation mode
    model.eval()

    end = time.time()
    with torch.no_grad():
      for i, (input_rgb, input_flow, target) in enumerate(val_loader):
        
          if use_gpu:
             input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
             input_flow = torch.autograd.Variable(input_flow.float().cuda())
             target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))
          else:
             input_rgb = torch.autograd.Variable(input_rgb.float())
             input_flow = torch.autograd.Variable(input_flow.float())
             target = torch.autograd.Variable(target.long())

          # compute output
          output = model(input_rgb,input_flow)
          loss = criterion(output, target)

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
                     i, len(val_loader), batch_time=batch_time, loss=losses,
                     top1=top1, top5=top5))

      print(' Epoch:{0} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(epoch, top1=top1, top5=top5))

      results = open('cnn_valid.txt', 'a')
      results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
            .format(epoch, loss=losses, top1=top1, top5=top5))
      results.close()

      return top1.avg



if __name__ == '__main__':
   main()
