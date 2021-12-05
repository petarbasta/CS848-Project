from torchvision.models.resnet import ResNet, Bottleneck

resnet_layers = [3, 4, 6, 3]

class HorovodRayTuneResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        return super(HorovodRayTuneResNet50, self).__init__(
            Bottleneck, resnet_layers, *args, **kwargs)

    def forward(self, x):
        return super(HorovodRayTuneResNet50, self).forward(x)


def build_horovod_raytune_resnet():
    return HorovodRayTuneResNet50()
