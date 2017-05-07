#encoding:utf-8

import torch
import csv, os, time
import argparse
import os.path
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.parallel
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from data_utils import AverageMeter, TestImageFolder
from PIL import Image

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='Pytorch Cats vs Dogs fine-tuning example')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch' ,metavar='ARCH', default='resnet101', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet101)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

best_prec1 = 0

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def accuracy(y_pred, y_actual, topk=(1, )):
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)
        
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))

        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))

        acc.update(prec1[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
    print(' * Accuracy {acc.avg:.3f}'.format(acc=acc))
    return acc.avg
def test(test_loader, model):
    csv_map = {}
    model.eval()
    for i, (images, filepath) in enumerate(test_loader):
        filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
        filepath = int(filepath)
        image_var = torch.autograd.Variable(images, volatile=True)
        y_pred = model(image_var)
        smax = nn.Softmax()
        smax_out = smax(y_pred)[0]
        cat_prob = smax_out.data[0]
        dog_prob = smax_out.data[1]
        prob = dog_prob
        if cat_prob > dog_prob:
            prob = 1 - cat_prob
        prob = np.around(prob, decimals=4)
        prob = np.clip(prob, .0001, .999)
        csv_map[filepath] = prob
    with open(os.path.join(args.data, 'entry.csv'), 'wb') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, 2)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # how to get these numbers ?

    train_loader = data.DataLoader(dsets.ImageFolder(traindir, transforms.Compose([transforms.RandomSizedCrop(224), 
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])), 
                                    batch_size = args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = data.DataLoader(dsets.ImageFolder(valdir, transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224),
                                    transforms.ToTensor(), normalize])), 
                                    batch_size = args.batch_size, shuffle=False, num_workers = 1, pin_memory=True)
    test_loader = data.DataLoader(TestImageFolder(testdir,transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224),
                                    transforms.ToTensor(), normalize,])),
                                    batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    if args.test:
        print("Testing the model and generating a output csv for submission")
        test(test_loader, model)
        exit(0)                                    
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.Adam(model.module.fc.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        exit(0)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch':epoch+1, 'arch':args.arch, 'state_dict':model.state_dict(),'best_prec1':best_prec1}, is_best)

