"""

This code repository is an official implementation of the paper:
Convolutional Transformer Fusion Blocks (CTFB) for Multi-Modal Gesture Recognition

Test script of the method Convolutional Transformer Fusion on NVGesture dataset

"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.i3dpt import I3D
from models.fusion_i3d import ConvTransformerFusion
from dataset.nvdataset import test_dataset
from utils import accuracy, AverageMeter

import time
import copy
import random
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

seed = 52353543
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

print_freq=10
use_gpu = torch.cuda.is_available()

def load_weights(args):
      
    i3d = I3D(num_classes=400, modality='rgb')
    i3d.conv3d_0c_1x1 = nn.Linear(1024, args.num_classes)

    return i3d

def main(args):

    i3d_rgb = load_weights(args)
    i3d_depth = load_weights(args)   

    model = ConvTransformerFusion(copy.deepcopy(i3d_rgb), copy.deepcopy(i3d_depth), args.num_classes)
    if args.use_dataparallel:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_checkpoint))
    

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()                         
    cudnn.benchmark = True
                                                                                                                                                                    
    tst_dataset = test_dataset(args.datadir, args.lst_test_file, args.seg_len, False)                                                                                                                                                                    
    test_loader =  torch.utils.data.DataLoader(tst_dataset, batch_size=args.batchsize,
                                               shuffle=True, num_workers=args.num_workers)

    start_time= time.time()
    epoch=0
    prec1 = validate(test_loader, model, criterion, epoch)
    print('Test Accuracy:', prec1)

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
             target = torch.autograd.Variable(target.long().cuda())
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

      results = open('./logs/test_log.txt', 'a')
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
      plt.savefig('nv_confusion.pdf')
      
      return top1.avg

def parse_args():
  parser = argparse.ArgumentParser(description='Modality optimization.')
  parser.add_argument('--load_checkpoint', type=str, help='load best model', default='./load_checkpoint/nv_fusion_best_87_76.pt')
  parser.add_argument('--datadir', type=str, help='root path of data directory', default='/path/to/datadir')
  parser.add_argument('--lst_test_file', type=str, help='list of test files', default='./meta/nvgesture_test_correct.lst')
  parser.add_argument('--num_classes', type=int, help='output dimension', default=25)
  parser.add_argument('--batchsize', type=int, help='batch size', default=4)
  parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel', default=False)
  parser.add_argument('--num_workers', dest='num_workers', type=int, help='Dataloader CPUS', default=8)
  parser.add_argument("--seg_len", action="store", default=64, dest="seg_len", type=int, nargs='+', help="length of video")

  return parser.parse_args()

if __name__ == '__main__':
   
   args = parse_args()
   main(args)

""" 
 Test run: 
 
 CUDA_VISIBLE_DEVICES=0 python3 test_two_stream.py --datadir '/storage/hampiholi/datasets/nvgesture' --load_checkpoint './load_checkpoint/nv_fusion_best_87_76.pt'
 
 """