import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
from torchsummary import summary

from models.i3dpt import I3D
from models.resnet import generate_model
from models.mobilenetv2 import MobileNetV2
from models.shufflenetv2 import ShuffleNetV2
from models.squeezenet import SqueezeNet

from dataset.depth import training_dataset, test_dataset
from utils import accuracy, AverageMeter
from config import *

import shutil
import time
import random
import numpy as np
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def save_checkpoint(state_dict, is_best, filename='./save_model/flow_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, path2best_model)


print_freq=10
use_gpu = torch.cuda.is_available()

def main():

    best_prec1=0
       
    #i3d_model = I3D(num_classes=400, modality='flow')
    #i3d_model.load_state_dict(torch.load('load_model/kinetics_mobilenet_1.0x_RGB_16_best.pth'))
    #i3d_model.conv3d_0c_1x1 = nn.Linear(1024, num_classes)

    #i3d_model = generate_model(model_depth=50, n_classes=700)
    #i3d_model.load_state_dict(torch.load('load_model/r3d50_K_200ep.pth')['state_dict'])
    #model = nn.Sequential(i3d_model, nn.ReLU(), nn.Linear(700, num_classes))
    #i3d_model.fc = nn.Linear(2048, num_classes)

    mv2 = MobileNetV2(num_classes=600, sample_size=224, width_mult=1.)
    mv2.classifier = nn.Linear(1280, num_classes)
    #mv2 = torch.nn.DataParallel(mv2).cuda()
    #mv2.load_state_dict(torch.load(path)) 

    #shuffle = SqueezeNet(sample_size=224, sample_duration=64, version=1.1, num_classes=600)
    #shuffle = ShuffleNetV2(num_classes=600, sample_size=112, width_mult=1.)
    #shuffle = torch.nn.DataParallel(shuffle).cuda()
    #model = nn.Sequential(shuffle, nn.ReLU(), nn.Linear(600, num_classes))
    #shuffle.load_state_dict(torch.load('load_model/nv_rgb_shuffle_best.pt'))

    #mv2.load_state_dict(torch.load('load_model/kinetics_squeezenet_RGB_16_best.pth')['state_dict'])
    #model = nn.Sequential(mv2, nn.ReLU(), nn.Linear(600, num_classes))
    model = mv2.cuda()
    summary(model, [(3,64,224,224)])
    exit(1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)   
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,25])                        
    cudnn.benchmark = True
    
    train_dataset = training_dataset(nvgesture_root, lst_train_file, 64, True)                                                                                                                                                                    
    valid_dataset = test_dataset(nvgesture_root, lst_val_file, 64, False)
                                                                                                                                                                          
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                               shuffle=True, num_workers=num_workers)

    val_loader =  torch.utils.data.DataLoader(valid_dataset, batch_size=test_batch_size,
                                               shuffle=True, num_workers=num_workers)

    start_time= time.time()

    for epoch in range(0, epochs):

        # train on train dataset
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
    for i, (input_rgb, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if use_gpu:
           input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
           target = torch.autograd.Variable(target.long().cuda())

        else:
           input_rgb = torch.autograd.Variable(input_rgb.float())
           target = torch.autograd.Variable(target.long())

        # compute output

        output = model(input_rgb)
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
      for i, (input_rgb, target) in enumerate(val_loader):
        
          if use_gpu:
             input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
             target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))
          else:
             input_rgb = torch.autograd.Variable(input_rgb.float())
             target = torch.autograd.Variable(target.long())          
          
          # compute output
          output = model(input_rgb)
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

      results = open('uni_valid.txt', 'a')
      results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
            .format(epoch, loss=losses, top1=top1, top5=top5))
      results.close()

      return top1.avg



if __name__ == '__main__':
   main()
