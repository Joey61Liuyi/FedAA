import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet
import numpy as np

class LowRank(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 low_rank: int,
                 kernel_size: int):
        super().__init__()
        self.T = nn.Parameter(
            torch.empty(size=(low_rank, low_rank, kernel_size, kernel_size)),
            requires_grad=True
        )
        self.O = nn.Parameter(
            torch.empty(size=(low_rank, out_channels)),
            requires_grad=True
        )
        self.I = nn.Parameter(
            torch.empty(size=(low_rank, in_channels)),
            requires_grad=True
        )
        self._init_parameters()

    def _init_parameters(self):
        # Initialization affects the convergence stability for our parameterization
        fan = nn.init._calculate_correct_fan(self.T, mode='fan_in')
        gain = nn.init.calculate_gain('relu', 0)
        std_t = gain / np.sqrt(fan)

        fan = nn.init._calculate_correct_fan(self.O, mode='fan_in')
        std_o = gain / np.sqrt(fan)

        fan = nn.init._calculate_correct_fan(self.I, mode='fan_in')
        std_i = gain / np.sqrt(fan)

        nn.init.normal_(self.T, 0, std_t)
        nn.init.normal_(self.O, 0, std_o)
        nn.init.normal_(self.I, 0, std_i)

    def forward(self):
        # torch.einsum simplify the tensor produce (matrix multiplication)
        return torch.einsum("xyzw,xo,yi->oizw", self.T, self.O, self.I)

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = False,
                 ratio: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.ratio = ratio
        self.low_rank = self._calc_from_ratio()

        self.W1 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
        self.W2 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _calc_from_ratio(self):
        # Return the low-rank of sub-matrices given the compression ratio
        r1 = int(np.ceil(np.sqrt(self.out_channels)))
        r2 = int(np.ceil(np.sqrt(self.in_channels)))
        r = np.max((r1, r2))

        num_target_params = self.out_channels * self.in_channels * \
                            (self.kernel_size ** 2) * self.ratio
        r3 = np.sqrt(
            ((self.out_channels + self.in_channels) ** 2) / (4 * (self.kernel_size ** 4)) + \
            num_target_params / (2 * (self.kernel_size ** 2))
        ) - (self.out_channels + self.in_channels) / (2 * (self.kernel_size ** 2))
        r3 = int(np.ceil(r3))
        r = np.max((r, r3))

        return r

    def forward(self, x):
        # Hadamard product of two submatrices
        W = self.W1() * self.W2()
        out = F.conv2d(input=x, weight=W, bias=self.bias,
                       stride=self.stride, padding=self.padding)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet
    Note two main differences from official pytorch version:
    1. conv1 kernel size: pytorch version uses kernel_size=7
    2. average pooling: pytorch version uses AdaptiveAvgPool
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feature_dim = 512 * block.expansion

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
