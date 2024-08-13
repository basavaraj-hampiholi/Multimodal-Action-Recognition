"""

This code repository is an official implementation of the paper:
Convolutional Transformer Fusion Blocks (CTFB) for Multi-Modal Gesture Recognition

This script performs training and validation of the method Convolutional Transformer Fusion on NVGesture dataset

"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.i3dpt import I3D
from models.fusion_i3d import ConvTransformerFusion
from dataset.nvdataset import training_dataset, test_dataset
from utils import accuracy, AverageMeter

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import time
import random
import copy
import shutil
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def save_checkpoint(state_dict, is_best, filename='./save_model/nvgesture_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, args.best_model)

use_gpu = torch.cuda.is_available()
print_freq=10


def load_weights(path, args):

    i3d = I3D(num_classes=400, modality='rgb')
    i3d.conv3d_0c_1x1 = nn.Linear(1024, args.num_classes)
    i3d.load_state_dict(torch.load(path))

    return i3d


def main(args):

    best_prec1=0
    i3d_rgb = load_weights(args.rgb_cp, args)
    i3d_depth = load_weights(args.depth_cp, args)
    
    model = ConvTransformerFusion(copy.deepcopy(i3d_rgb), copy.deepcopy(i3d_depth), args.num_classes)
    if args.use_dataparallel:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    #summary(model, [(3,64,224,224), (3,64,224,224)])

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.init_lr, weight_decay = args.weight_decay, momentum=0.9, nesterov=True)                         
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_scheduler_milestones, args.lr_drop_ratio)
    cudnn.benchmark = True

    transformed_train_dataset = training_dataset(args.datadir, args.lst_train_file, args.seg_len, True)                                                                                                                                                                                                                
    transformed_validation_dataset = test_dataset(args.datadir, args.lst_test_file, args.seg_len, False)                                                                                     
                                                                                                                                
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(transformed_train_dataset, batch_size=args.batchsize,
                                               shuffle=True, num_workers=args.num_workers)

    val_loader =  torch.utils.data.DataLoader(transformed_validation_dataset, batch_size=args.batchsize,
                                               shuffle=True, num_workers=args.num_workers)

    start_time= time.time()

    writer = SummaryWriter()

    for epoch in range(0, args.epochs):
        # train on train dataset
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)
        print('Top Precision:',prec1)

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model.state_dict(), is_best, args.checkpointdir)

    end_time = time.time()
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)
    writer.flush()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer):
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
           target = torch.autograd.Variable(target.long().cuda())

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
  
        writer.add_scalar("Loss/train", losses.avg, epoch)
        writer.add_scalar("Prec1/train", top1.avg, epoch)

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

    results = open('./logs/train_log.txt', 'a')
    results.write('Epoch: [{0}][{1}/{2}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
           epoch, i, len(train_loader), loss=losses,
           top1=top1, top5=top5))
    results.close()

    

def validate(val_loader, model, criterion, epoch, writer):
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
             target = torch.autograd.Variable(target.long().cuda())
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

          writer.add_scalar("Loss/test", losses.avg, epoch)
          writer.add_scalar("Prec1/test", top1.avg, epoch)
      
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

      results = open('./logs/validation_log.txt', 'a')
      results.write('Epoch:{0} Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
            .format(epoch, loss=losses, top1=top1, top5=top5))
      results.close()

      return top1.avg


def parse_args():
  parser = argparse.ArgumentParser(description='Two stream fusion experiments on NVGesture dataset')
  parser.add_argument('--checkpointdir', type=str, help='save checkpoint dir', default='./save_checkpoint/nv_fusion_checkpoint.pt')
  parser.add_argument('--best_model', type=str, help='save best model', default='./save_checkpoint/nv_fusion_best.pt')
  parser.add_argument('--datadir', type=str, help='data directory', default='/path/to/data/root/directory')
  parser.add_argument('--lst_train_file', type=str, help='list of train files', default='./meta/nvgesture_train_correct.lst')
  parser.add_argument('--lst_test_file', type=str, help='list of test files', default='./meta/nvgesture_test_correct.lst')
  parser.add_argument('--rgb_cp', type=str, help='RGB i3d checkpoint', default='./load_checkpoint/nv_rgb_best.pt')
  parser.add_argument('--depth_cp', type=str, help='depth i3d checkpoint', default='./load_checkpoint/nv_depth_best.pt')
  parser.add_argument('--num_classes', type=int, help='number of classes ', default=25)
  parser.add_argument('--batchsize', type=int, help='batch size', default=4)
  parser.add_argument('--epochs', type=int, help='number of training epochs', default=30)
  parser.add_argument('--init_lr', type=int, help='initial learning rate', default=0.005)
  parser.add_argument('--lr_drop_ratio', type=int, help='drop learning rate', default=0.1)
  parser.add_argument('--lr_scheduler_milestones', type=int, help='drop learning rate at specific epochs', default=[15,25])
  parser.add_argument('--weight_decay', type=int, help='L2 regularization ', default=1e-4)
  parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel', default=False)
  parser.add_argument('--num_workers', dest='num_workers', type=int, help='Dataloader CPUS', default=8)
  parser.add_argument("--seg_len", action="store", default=64, dest="seg_len", type=int, nargs='+', help="length of video")

  return parser.parse_args()

if __name__ == '__main__':
   
   args = parse_args()
   main(args)

# Example run: CUDA_VISIBLE_DEVICES=0 python3 train_two_stream.py --datadir '/storage/hampiholi/datasets/nvgesture'   