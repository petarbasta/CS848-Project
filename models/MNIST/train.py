"""
A script for training various torchvision models on MNIST data.
This is a modified version of the MNIST training example script from the PyTorch team.

Original script: https://github.com/pytorch/examples/tree/master/mnist
Retrieval date: Dec 7, 2021

python train.py --arch resnet --parallelism dp

Note 1: Data Parallelism only supports single node / multiple GPU training.
Note 2: You can specify hyperparameters using other available arguments.
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from timeit import default_timer as timer
import torch.distributed as dist
import json
import torch.multiprocessing as mp

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
from torchvision.models.resnet import ResNet
from parallel_models import (
    build_dp_resnet,
    build_mp_resnet,
    build_gpipe_resnet,
    build_dp_alexnet,
    build_mp_alexnet,
    build_gpipe_alexnet
)      


assert torch.cuda.is_available(), "CUDA must be available in order to run"
n_gpus = torch.cuda.device_count()
assert n_gpus == 2, f"MNIST training requires exactly 2 GPUs to run, but got {n_gpus}"

supported_model_architectures = ['resnet', 'alexnet']
supported_parallelism_strategies = ['dp', 'mp', 'gpipe']
supported_models = {
    'resnet': {
        'dp': build_dp_resnet,
        'mp': build_mp_resnet,
        'gpipe': build_gpipe_resnet,
    },
    'alexnet': {
        'dp': build_dp_alexnet,
        'mp': build_mp_alexnet,
        'gpipe': build_gpipe_alexnet
    }
}

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if args.parallelism == 'dp':
            data = data.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        elif args.parallelism == 'mp':
            data = data.cuda(1, non_blocking=True)
            target = target.cuda(1, non_blocking=True)
        elif args.parallelism == 'gpipe':
            data = data.to(model.devices[0], non_blocking=True)
            target = target.to(model.devices[-1], non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            if args.parallelism == 'dp':
                data = data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            elif args.parallelism == 'mp':
                data = data.cuda(1, non_blocking=True)
                target = target.cuda(1, non_blocking=True)
            elif args.parallelism == 'gpipe':
                data = data.to(model.devices[0], non_blocking=True)
                target = target.to(model.devices[-1], non_blocking=True)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def init_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--parallelism', default='dp',
                        choices=supported_parallelism_strategies,
                        help='training parallelism strategy: ' +
                            ' | '.join(supported_parallelism_strategies) +
                            ' (default: dp)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=supported_model_architectures,
                        help='model architecture: ' +
                            ' | '.join(supported_model_architectures) +
                            ' (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Initialize dp-specific config
    args.gpu = None
    args.world_size = 1
    args.rank = 0
    args.dist_url = 'tcp://127.0.0.1:8001'
    args.dist_backend = 'nccl'

    return args


def main():
    args = init_args()
    torch.manual_seed(args.seed)
    ngpus_per_node = torch.cuda.device_count()

    manager = mp.Manager()
    best_accuracy = manager.Value('d', 0)
    mem_params = manager.Value('d', 0)
    mem_bufs = manager.Value('d', 0)
    mem_peak = manager.Value('d', 0)

    start_time = timer()

    if args.parallelism == "dp":
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak))
    else:
        main_worker(args.gpu, ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak)

    end_time = timer()

    reported_stats = {'accuracy': best_accuracy.value, 'runtime': end_time - start_time, 'mem_params': mem_params.value, 'mem_bufs': mem_bufs.value, 'mem_peak': mem_peak.value }
    print(json.dumps(reported_stats))

    
def main_worker(gpu, ngpus_per_node, args, best_accuracy, mem_params, mem_bufs, mem_peak):
    args.gpu = gpu

    # Initialize process group
    if args.parallelism == 'dp':
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                world_size=args.world_size, rank=args.rank)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    # if use_cuda
    if True:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = supported_models[args.arch][args.parallelism]()

    if args.parallelism == 'gpipe':
        partitions = torch.cuda.device_count()
        sample = torch.rand(64, 1, 28, 28)
        balance = balance_by_time(partitions, model, sample)
        model = GPipe(model, balance, chunks=8)
    elif args.parallelism =='dp':
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        accuracy = test(model, test_loader, args)
        scheduler.step()

        # Update best accuracy across all epochs
        best_accuracy.value = max(accuracy, best_accuracy.value)

    mem_params.value = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs.value = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem_peak.value = torch.cuda.max_memory_allocated()
    #if args.save_model:
    #    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

