import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.alexnet import AlexNet
from collections import OrderedDict
from typing import Any, List
from gpipe_bottleneck import bottleneck
from gpipe_flatten_sequential import flatten_sequential

# NOTE: We assume host machine has 2 GPUs
assert torch.cuda.is_available(), "CUDA must be available in order to run"
n_gpus = torch.cuda.device_count()
assert n_gpus == 2, f"MP ResNet requires exactly 2 GPUs to run, but got {n_gpus}"

resnet_layers = [3, 4, 6, 3]

class DataParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(DataParallelResNet50, self).__init__(
            Bottleneck, resnet_layers, *args, **kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return super(DataParallelResNet50, self).forward(x)

class DataParallelAlexNet(AlexNet):
    def __init__(self, *args, **kwargs):
        super(DataParallelAlexNet, self).__init__(*args, **kwargs)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        return super(DataParallelAlexNet, self).forward(x)

def build_dp_resnet():
    return DataParallelResNet50()

def build_dp_alexnet():
    return DataParallelAlexNet()


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, resnet_layers, *args, **kwargs)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = x.to('cuda:0')
        x = self.seq1(x)
        x = x.to('cuda:1')
        x = self.seq2(x)
        x = x.view(x.size(0), -1).to('cuda:1')
        x = self.fc(x)
        return x

class ModelParallelAlexNet(AlexNet):
    def __init__(self, num_classes: int = 1000) -> None:
        super(ModelParallelAlexNet, self).__init__()
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
        ).to('cuda:0')

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)).to('cuda:1')

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ).to('cuda:1')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to('cuda:0')
        x = self.features(x)
        x = x.to('cuda:1')
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_mp_resnet():
    return ModelParallelResNet50()

def build_mp_alexnet(num_classes: int = 1000):
    return ModelParallelAlexNet(num_classes = num_classes)

# pytorchgpipe compatible ResNet rewritten as nn.Sequential
# Source: https://github.com/kakaobrain/torchgpipe/blob/master/benchmarks/models/resnet/
# Retrieved on: Nov 27, 2021

def _build_sequential_resnet(layers: List[int],
                 num_classes: int = 1000,
                 inplace: bool = False
                 ) -> nn.Sequential:
    """Builds a ResNet as a simple sequential model.

    Note:
        The implementation is copied from :mod:`torchvision.models.resnet`.

    """
    inplanes = 64

    def make_layer(planes: int,
                   blocks: int,
                   stride: int = 1,
                   inplace: bool = False,
                   ) -> nn.Sequential:
        nonlocal inplanes

        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(bottleneck(inplanes, planes, stride, downsample, inplace))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(bottleneck(inplanes, planes, inplace=inplace))

        return nn.Sequential(*layers)

    # Build ResNet as a sequential model.
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', make_layer(64, layers[0], inplace=inplace)),
        ('layer2', make_layer(128, layers[1], stride=2, inplace=inplace)),
        ('layer3', make_layer(256, layers[2], stride=2, inplace=inplace)),
        ('layer4', make_layer(512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * 4, num_classes)),
    ]))

    # Flatten nested sequentials.
    model = flatten_sequential(model)

    # Initialize weights for Conv2d and BatchNorm2d layers.
    # Stolen from torchvision-0.4.0.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return

    model.apply(init_weight)

    return model


def build_gpipe_resnet(**kwargs: Any) -> nn.Sequential:
    return _build_sequential_resnet(resnet_layers, **kwargs)

def build_gpipe_alexnet(**kwargs: Any) -> nn.Sequential:
    return AlexNet()
