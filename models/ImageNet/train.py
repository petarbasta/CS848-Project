"""
A script for training various torchvision models on ImageNet data.
This is a modified version of the ImageNet training example script from the PyTorch team.

Original script: https://github.com/pytorch/examples/tree/master/imagenet
Retrieval date: Nov 26, 2021

python train.py -a resnet --parallelism dp /u4/jerorseth/datasets/ILSVRC/Data/CLS-LOC

Note 1: Data Parallelism (and Hybrid) only supports single node / multiple GPU training.
Note 2: You can specify hyperparameters using other available arguments.
"""

import argparse
import os
import random
import shutil
import time
import json
import warnings
from enum import Enum
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time, balance_by_size
from torchvision.models.resnet import ResNet
from parallel_models import (
    build_dp_resnet,
    build_mp_resnet,
    build_gpipe_resnet,
    build_dp_alexnet,
    build_mp_alexnet      
)

assert torch.cuda.is_available(), "CUDA must be available in order to run"
n_gpus = torch.cuda.device_count()
assert n_gpus == 2, "ImageNet training requires exactly 2 GPUs to run, but got {n_gpus}"

supported_model_architectures = ['resnet']
supported_parallelism_strategies = ['dp', 'mp', 'gpipe']
supported_models = {
    'resnet': {
        'dp': build_dp_resnet,
        'mp': build_mp_resnet,
        'gpipe': build_gpipe_resnet,
    },
    'alexnet': {
        'dp': build_dp_alexnet,
        'mp': build_mp_alexnet
    }
}


def init_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-d', '--data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=supported_model_architectures,
                        help='model architecture: ' +
                            ' | '.join(supported_model_architectures) +
                            ' (default: resnet)')
    parser.add_argument('--parallelism', default='dp',
                        choices=supported_parallelism_strategies,
                        help='training parallelism strategy: ' +
                            ' | '.join(supported_parallelism_strategies) +
                            ' (default: dp)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use.')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                         'N processes per node, which has N GPUs. This is the '
    #                         'fastest way to use PyTorch for either single node or '
    #                         'multi node data parallel training')
    
    # Manually override expected arguments to setup single-node multi-GPU distributed training
    # This should only occur when Data Parallelism (mp) or Hybrid Parallelism (hp) is specified
    args = parser.parse_args()
    train_distributed = args.parallelism == 'dp'

    args.pretrained = None
    args.gpu = None
    args.world_size = 1 if train_distributed else -1
    args.rank = 0 if train_distributed else -1
    args.dist_url = 'tcp://127.0.0.1:8001' if train_distributed else 'tcp://224.66.41.62:23456'
    args.dist_backend = 'nccl'
    args.multiprocessing_distributed = True if train_distributed else None
    return args


def main():
    args = init_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    manager = mp.Manager()
    best_accuracy = manager.Value('d', 0)
    mem_params = manager.Value('d', 0)
    mem_bufs = manager.Value('d', 0)
    mem_peak = manager.Value('d', 0)

    start_time = timer()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak)
    end_time = timer()

    # After workers have completed, output statistics
    reported_stats = {
        'accuracy': best_accuracy.value,
        'runtime': end_time - start_time,
        'mem_params': mem_params.value,
        'mem_bufs': mem_bufs.value,
        'mem_peak': mem_peak.value }

    print(json.dumps(reported_stats))


def main_worker(gpu, ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = supported_models[args.arch][args.parallelism]()

    if args.parallelism == 'gpipe':
        partitions = torch.cuda.device_count()
        sample = torch.rand(128, 3, 224, 224)
        balance = balance_by_time(partitions, model, sample)
        model = GPipe(model, balance, chunks=8)
    elif args.parallelism == 'hp':
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        # If hp (ie. mp + dp) is specified, then device_ids and output_device must NOT be set!
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.distributed: # dp
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    if args.parallelism == 'gpipe':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    """
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        accuracy_num = acc1.item() if torch.is_tensor(acc1) else float(acc1)
        # is_best = accuracy_num > best_accuracy.value
        best_accuracy.value = max(accuracy_num, best_accuracy.value)

    mem_params.value = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs.value = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    # TODO: Not sure whether this will work when no GPUs are found, checking to be safe
    if args.parallelism != 'none':
        mem_peak.value = torch.cuda.max_memory_allocated()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.parallelism == 'dp':
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        elif args.parallelism == 'mp':
            images = images.cuda(1, non_blocking=True)
            target = target.cuda(1, non_blocking=True)
        elif args.parallelism == 'gpipe':
            images = images.to(model.devices[0], non_blocking=True)
            target = target.to(model.devices[-1], non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.parallelism == 'dp':
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            elif args.parallelism == 'mp':
                images = images.cuda(1, non_blocking=True)
                target = target.cuda(1, non_blocking=True)
            elif args.parallelism == 'gpipe':
                images = images.to(model.devices[0], non_blocking=True)
                target = target.to(model.devices[-1], non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


