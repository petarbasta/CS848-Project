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

# TODO: These might be able to be imported conditionally
import horovod.torch as hvd
from ray import tune

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
from parallel_models import (
    build_dp_resnet,
    build_mp_resnet,
    build_gpipe_resnet,
    build_dp_alexnet,
    build_mp_alexnet,
    build_gpipe_alexnet,
    build_horovod_raytune_resnet,
    build_horovod_raytune_alexnet,
    build_basic_alexnet,
    build_basic_resnet
)

supported_model_architectures = ['resnet', 'alexnet']
supported_parallelism_strategies = ['none', 'dp', 'mp', 'gpipe']
supported_models = {
    'resnet': {
        'none': build_basic_resnet,
        'horovod_raytune': build_horovod_raytune_resnet,
        'dp': build_dp_resnet,
        'mp': build_mp_resnet,
        'gpipe': build_gpipe_resnet,
    },
    'alexnet': {
        'none': build_basic_alexnet,
        'horovod_raytune': build_horovod_raytune_alexnet,
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
        elif args.parallelism == 'horovod_raytune':
            data, target = data.cuda(), target.cuda()

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
            elif args.parallelism == 'horovod_raytune':
                data, target = data.cuda(), target.cuda()

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def init_hypertune_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--parallelism', default='none',
                        choices=supported_parallelism_strategies,
                        help='training parallelism strategy: ' +
                            ' | '.join(supported_parallelism_strategies) +
                            ' (default: none)')
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
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    args = parser.parse_args()

    # Initialize dp-specific config
    args.gpu = None
    args.world_size = 1
    args.rank = 0
    args.dist_url = 'tcp://127.0.0.1:8001'
    args.dist_backend = 'nccl'

    return args


def init_horovod_raytune_args(config, arch, epochs, dry_run):
    def get_arg(key, default_val):
        return config[key] if key in config else default_val
    
    args = type('', (), {})()
    args.parallelism = 'horovod_raytune'
    args.arch = arch

    # Hyperparameters are provided via config
    args.batch_size = get_arg('batch-size', 64)
    args.test_batch_size = get_arg('test-batch-size', 1000)
    args.lr = get_arg('lr', 1.0)
    args.gamma = get_arg('gamma', 0.7)

    # Regular args are passed to this init function
    args.epochs = epochs
    args.dry_run = dry_run

    # TODO: Either pass args for these args, or remove them from the script entirely
    args.log_interval = 10
    args.seed = 1
    args.gpu = None

    return args


def run_horovod_raytune_mnist_training(config, checkpoint_dir=None, arch=None, epochs=None, dry_run=None):
    assert torch.cuda.is_available(), "CUDA must be available in order to run"
    n_gpus = torch.cuda.device_count()
    assert n_gpus == 2, f"Horovod RayTune MNIST training requires exactly 2 GPUs to run, but got {n_gpus}"

    assert arch is not None, "arch is not specified"
    assert arch is not None, "epochs is not specified"
    assert dry_run is not None, "dry_run is not specified"

    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    args = init_horovod_raytune_args(config, arch, epochs, dry_run)
    main(args)


def run_hypertune_mnist_training():
    args = init_hypertune_args()
    if args.parallelism != 'none':
        assert torch.cuda.is_available(), "CUDA must be available in order to run"
        n_gpus = torch.cuda.device_count()
        assert n_gpus == 2, f"HyperTune MNIST training requires exactly 2 GPUs to run, but got {n_gpus}"

    main(args)


def main(args):
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

    reported_stats = {
        'accuracy': best_accuracy.value,
        'runtime': end_time - start_time,
        'mem_params': mem_params.value,
        'mem_bufs': mem_bufs.value,
        'mem_peak': mem_peak.value }
    
    if args.parallelism == 'horovod_raytune':
        tune.report(
            accuracy=reported_stats['accuracy'],
            runtime=reported_stats['runtime'],
            mem_peak=reported_stats['mem_peak'],
            mem_params=reported_stats['mem_params'],
            mem_bufs=reported_stats['mem_bufs'],
            rank=hvd.rank())
    
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

    if args.parallelism != 'none':
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

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # TODO: This could be refactored to use DistributedSampler
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Instantiate the DNN model
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
    elif args.parallelism == 'horovod_raytune':
        model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # If applicable, wrap optimizer in Horovod Distributed Optimizer
    if args.parallelism == 'horovod_raytune':
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        accuracy = test(model, test_loader, args)
        scheduler.step()

        best_accuracy.value = max(accuracy, best_accuracy.value)

    mem_params.value = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs.value = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    # TODO: Not sure whether this will work when no GPUs are found, checking to be safe
    if args.parallelism != 'none':
        mem_peak.value = torch.cuda.max_memory_allocated()

    #if args.save_model:
    #    torch.save(model.state_dict(), "mnist_cnn.pt")


# When this script is run on its own, assume HyperTune training
if __name__ == '__main__':
    run_hypertune_mnist_training()

