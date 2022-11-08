import torch.nn as nn
import torch
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,
                         stride=2),

            nn.Conv2d(in_channels=64,
                      out_channels=192,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),

            nn.Conv2d(in_channels=192,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.space_transformer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = 512
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # forward: forward propagation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.space_transformer(x)
        x = self.fc(x)
        return x

class MobileNet_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.fc = nn.Linear(1024, num_classes)
        self.feature_dim = 1024

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNet_Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out