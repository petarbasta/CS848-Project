"""
A script for training various torchvision models on MNIST data.
This is a modified version of the MNIST training example script from the PyTorch team.

Original script: https://github.com/pytorch/examples/tree/master/mnist
Retrieval date: Dec 7, 2021
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

assert torch.cuda.is_available(), "CUDA must be available in order to run"
n_gpus = torch.cuda.device_count()
assert n_gpus == 2, f"MNIST training requires exactly 2 GPUs to run, but got {n_gpus}"

class Flatten(nn.Sequentail):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class GPipeAlexNet(nn.Sequential):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.model = nn.Sequential(*(list(self.features.children()) + [self.avgpool, Flatten()] + list(self.classifier.children())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.to(model.devices[0], non_blocking=True)
        target = target.to(model.devices[-1], non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

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


def main():

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 64}

    cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = GPipeAlexNet()

    partitions = torch.cuda.device_count()
    sample = torch.rand(64, 1, 227, 227)
    balance = balance_by_time(partitions, model, sample)
    model = GPipe(model, balance, chunks=8)

    optimizer = optim.Adadelta(model.parameters(), lr=0.9)

    best_accuracy = 0

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 2):
        train(model, train_loader, optimizer, epoch)
        accuracy = test(model, test_loader)
        scheduler.step()

        best_accuracy = max(accuracy, best_accuracy)

    print(best_accuracy)

if __name__ == '__main__':
    main()

