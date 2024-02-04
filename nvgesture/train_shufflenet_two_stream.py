import shutil
from config_beide import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models.shufflenetv2 import ShuffleNetV2
from models.fusion_shuffle import TransFusion
from dataset.nvdataset import training_dataset, test_dataset
from utils import accuracy, AverageMeter
from torchsummary import summary
import time
import random
import numpy as np
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def save_checkpoint(state_dict, is_best, filename='./save_model/nvgesture_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, trained_model_path_best)

use_gpu = torch.cuda.is_available()
print_freq=10


def load_weights(path, type):
    if type=='rgb':
        shuffle = ShuffleNetV2(num_classes=600, sample_size=224, width_mult=1.)
        shuffle = torch.nn.DataParallel(shuffle).cuda()
        shuffle.module.classifier = nn.Linear(1024, num_classes)
        shuffle.load_state_dict(torch.load(path))

    elif type=='depth':
        shuffle = ShuffleNetV2(num_classes=600, sample_size=224, width_mult=1.)
        shuffle = torch.nn.DataParallel(shuffle).cuda()
        shuffle.module.classifier = nn.Linear(1024, num_classes)
        shuffle.load_state_dict(torch.load(path))  

    return shuffle 

def main():

    best_prec1=0
    i3d_rgb = load_weights('load_model/nv_rgb_shuffle_best.pt', 'rgb')
    i3d_flow = load_weights('load_model/nv_depth_shuffle_best.pt', 'depth')

    # for param in i3d_rgb.parameters():
    #  param.requires_grad = False   
    # for param in i3d_flow.parameters():
    #  param.requires_grad = False          

    model = TransFusion(copy.deepcopy(i3d_rgb), copy.deepcopy(i3d_flow), num_classes)
    #model = EarlyFusion(copy.deepcopy(i3d_rgb), num_classes)  
    model = model.cuda() #torch.nn.DataParallel(model).cuda()

    #model.load_state_dict(torch.load('load_model/nv_fusion_best.pt'))

    summary(model, [(3,64,224,224), (3,64,224,224)])
    exit(1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay = weight_decay, momentum=0.9, nesterov=True)                         
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,25])
    cudnn.benchmark = True

    transformed_train_dataset = training_dataset(nvgesture_root, lst_train_file, 64, True)                                                                                                                                                                                                                
    transformed_valid_dataset = test_dataset(nvgesture_root, lst_val_file, 64, False)                                                                                     
                                                                                                                                
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(transformed_train_dataset, batch_size=train_batch_size,
                                               shuffle=True, num_workers=num_workers)

    val_loader =  torch.utils.data.DataLoader(transformed_valid_dataset, batch_size=test_batch_size,
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
        save_checkpoint(model.state_dict(), is_best, trained_model_path)

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
    for i, (input_rgb, input_depth, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if use_gpu:
           input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
           input_depth = torch.autograd.Variable(input_depth.float().cuda())
           target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))

        else:
           input_rgb = torch.autograd.Variable(input_rgb.float())
           input_depth = torch.autograd.Variable(input_depth.float())
           target = torch.autograd.Variable(target.long())

        
        # compute output
        output = model(input_rgb, input_depth)
        loss = criterion(output, target)


        #measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data, input_rgb.size(0))
        top1.update(prec1, input_rgb.size(0))
        top5.update(prec5, input_rgb.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip)
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
      for i, (input_rgb, input_depth, target) in enumerate(val_loader):

          if use_gpu:
             input_rgb = torch.autograd.Variable(input_rgb.float().cuda())
             input_depth = torch.autograd.Variable(input_depth.float().cuda())
             target = torch.autograd.Variable(target.long().cuda())#.to('cuda:0'))
          else:
             input_rgb = torch.autograd.Variable(input_rgb.float())
             input_depth = torch.autograd.Variable(input_depth.float())
             target = torch.autograd.Variable(target.long())


          # compute output
          output = model(input_rgb, input_depth)
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