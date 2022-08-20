"""
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print('self.conv1',self.conv1.weight.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.avgpool(out3)
        out4 = x.view(x.size(0), -1)
        x = self.fc(out4)

        return x, [out1, out2, out3, out4]


class PreAct_ResNet_Cifar_no_bias(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_no_bias, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes, bias = False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        x = self.fc(out4)

        return x, [out1, out2, out3, out4]


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        x = self.fc(out4)

        return x, [out1, out2, out3, out4]


class PreAct_ResNet_Cifar_fm(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_fm, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.bn(out3)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)#
        out4 = x.view(x.size(0), -1)
        x = self.fc(out4)

        return x, [out1, out2, out3, out4]

class PreAct_ResNet_Cifar_Shared(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Shared, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.bn = nn.BatchNorm2d(64*block.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # out3 = self.layer3(out2)
        #
        # x = self.bn(out3)
        # x = self.relu(x)
        # x = self.avgpool(x)
        # out4 = x.view(x.size(0), -1)
        # x = self.fc(out4)

        return out2, [out1, out2]


class PreAct_ResNet_Cifar_Multi(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        # self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        # self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        # self.relu_p = nn.ReLU(inplace=True)
        # self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        x_g = self.fc(out4)


        # print('self.layer3_g')
        # out3_p = self.layer3_p(out2.detach())
        #
        # x_p  = self.bn_p(out3_p )
        # x_p  = self.relu_p(x_p )
        # x_p  = self.avgpool_p(x_p )
        # out4_p  = x_p.view(x_p.size(0), -1)
        x_p = self.fc_p(out4.detach())

        return [x_p, x_g], [out1, out2, out3, out4]


class PreAct_ResNet_Cifar_Multi_fm(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi_fm, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        # self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        # self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        # self.relu_p = nn.ReLU(inplace=True)
        # self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        x = self.bn(out3)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)#self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        x_g = self.fc(out4)


        # print('self.layer3_g')
        # out3_p = self.layer3_p(out2.detach())
        #
        # x_p  = self.bn_p(out3_p )
        # x_p  = self.relu_p(x_p )
        # x_p  = self.avgpool_p(x_p )
        # out4_p  = x_p.view(x_p.size(0), -1)
        x_p = self.fc_p(out4.detach())

        return [x_p, x_g], [out1, out2, out3, out4]

class PreAct_ResNet_Cifar_Multi_Proto(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi_Proto, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes,bias=False)

        self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        self.relu_p = nn.ReLU(inplace=True)
        self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        out4 = F.normalize(out4, dim=1)
        self.fc.weight = nn.Parameter(F.normalize(self.fc.weight, dim=1))
        x_g = self.fc(out4)


        # print('self.layer3_g')
        out3_p = self.layer3_p(out2.detach())

        x_p  = self.bn_p(out3_p )
        x_p  = self.relu_p(x_p )
        x_p  = self.avgpool_p(x_p )
        out4_p  = x_p.view(x_p.size(0), -1)
        out4_p = F.normalize(out4_p, dim=1)
        self.fc_p.weight = nn.Parameter(F.normalize(self.fc_p.weight, dim=1))
        x_p = self.fc_p(out4_p)

        return [x_p, x_g], [out1, out2, out3, out4]


class PreAct_ResNet_Cifar_Multi_Proto_v2(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi_Proto_v2, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes,bias=False)

        self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        self.relu_p = nn.ReLU(inplace=True)
        self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        out4 = F.normalize(out4, dim=1)
        self.fc.weight = nn.Parameter(F.normalize(self.fc.weight, dim=1))
        x_g = self.fc(out4)


        # # print('self.layer3_g')
        # out3_p = self.layer3_p(out2.detach())
        #
        # x_p  = self.bn_p(out3_p )
        # x_p  = self.relu_p(x_p )
        # x_p  = self.avgpool_p(x_p )
        # out4_p  = x_p.view(x_p.size(0), -1)
        # out4_p = F.normalize(out4_p, dim=1)
        self.fc_p.weight = nn.Parameter(F.normalize(self.fc_p.weight, dim=1))
        x_p = self.fc_p(out4.detach())

        return [x_p, x_g], [out1, out2, out3, out4]

class PreAct_ResNet_Cifar_Multi_Selector(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi_Selector, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        self.relu_p = nn.ReLU(inplace=True)
        self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes)

        self.layer3_selector = copy.deepcopy(self.layer3)
        self.bn_s = nn.BatchNorm2d(64 * block.expansion)
        self.relu_s = nn.ReLU(inplace=True)
        self.avgpool_s = nn.AvgPool2d(8, stride=1)
        self.fc_s = nn.Linear(64 * block.expansion, 2)
        self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        out3 = self.bn(out3)
        out3 = self.relu(out3)
        out3 = self.avgpool(out3)
        out4 = out3.view(out3.size(0), -1)
        x_g = self.fc(out4)


        # print('self.layer3_g')
        # out3_p = self.layer3_p(out2.detach())

        # x_p  = self.bn_p(out3_p )
        # x_p  = self.relu_p(x_p )
        # x_p  = self.avgpool_p(x_p )
        # out4_p  = x_p.view(x_p.size(0), -1)
        x_p = self.fc_p(out4.detach())

        # x_s = self.layer3_selector(out2.detach())
        # # print('x_s',x_s.size())
        # x_s = self.bn_s(x_s)
        # x_s = self.relu_s(x_s)
        # x_s = self.avgpool_s(x_s)
        # out4_s = x_s.view(x_s.size(0), -1)
        x_s = self.fc_s(out4.detach())
        x_s = self.sigmoid(x_s)



        return [x_p, x_g, x_s], [out1, out2, out3, out4]


class PreAct_ResNet_Cifar_Multi_Selector_no_sigmoid(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar_Multi_Selector_no_sigmoid, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        self.layer3_p = copy.deepcopy(self.layer3) #self._make_layer(block, 64, layers[2], stride=2)
        self.bn_p = nn.BatchNorm2d(64 * block.expansion)
        self.relu_p = nn.ReLU(inplace=True)
        self.avgpool_p = nn.AvgPool2d(8, stride=1)
        self.fc_p = nn.Linear(64 * block.expansion, num_classes)

        self.sigma1 = nn.Parameter(torch.zeros(1))
        self.sigma2 = nn.Parameter(torch.zeros(1))

        # self.layer3_selector = copy.deepcopy(self.layer3)
        # self.bn_s = nn.BatchNorm2d(64 * block.expansion)
        # self.relu_s = nn.ReLU(inplace=True)
        # self.avgpool_s = nn.AvgPool2d(8, stride=1)
        # self.fc_s = nn.Linear(64 * block.expansion, 2)
        # self.sigmoid = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        # print('out2',out2.size())
        # print('self.layer3')
        out3 = self.layer3(out2)
        # print('self.layer3',self.layer3[0].weight)

        out3 = self.bn(out3)
        out3 = self.relu(out3)
        out3 = self.avgpool(out3)
        out4 = out3.view(out3.size(0), -1)
        x_g = self.fc(out4)


        # print('self.layer3_g')
        # out3_p = self.layer3_p(out2.detach())

        # x_p  = self.bn_p(out3_p )
        # x_p  = self.relu_p(x_p )
        # x_p  = self.avgpool_p(x_p )
        # out4_p  = x_p.view(x_p.size(0), -1)
        x_p = self.fc_p(out4.detach())

        # x_s = self.layer3_selector(out2.detach())
        # # print('x_s',x_s.size())
        # x_s = self.bn_s(x_s)
        # x_s = self.relu_s(x_s)
        # x_s = self.avgpool_s(x_s)
        # out4_s = x_s.view(x_s.size(0), -1)
        # x_s = self.fc_s(out4.detach())
        # x_s = self.sigmoid(x_s)



        return [x_p, x_g, (self.sigma1, self.sigma2)], [out1, out2, out3, out4]

def resnet14_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
    return model

def resnet8_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
    return model


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet26_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [4, 4, 4], **kwargs)
    return model

def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model

def preact_resnet14_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
    return model

def preact_resnet8_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet20_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def preact_resnet8_cifar_shared(**kwargs):
    model = PreAct_ResNet_Cifar_Shared(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_multi(**kwargs):
    model = PreAct_ResNet_Cifar_Multi(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_multi_fm(**kwargs):
    model = PreAct_ResNet_Cifar_Multi_fm(BasicBlock, [1, 1, 1], **kwargs)
    return model


def preact_resnet8_cifar_fm(**kwargs):
    model = PreAct_ResNet_Cifar_fm(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_no_bias(**kwargs):
    model = PreAct_ResNet_Cifar_no_bias(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_multi_proto(**kwargs):
    model = PreAct_ResNet_Cifar_Multi_Proto(BasicBlock, [1, 1, 1], **kwargs)
    return model


def preact_resnet8_cifar_multi_proto_v2(**kwargs):
    model = PreAct_ResNet_Cifar_Multi_Proto_v2(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_multi_selector(**kwargs):
    model = PreAct_ResNet_Cifar_Multi_Selector(BasicBlock, [1, 1, 1], **kwargs)
    return model

def preact_resnet8_cifar_multi_selector_no_sigmoid(**kwargs):
    model = PreAct_ResNet_Cifar_Multi_Selector_no_sigmoid(BasicBlock, [1, 1, 1], **kwargs)
    return model


resnet_book = {
	'8': resnet8_cifar,
	'14': resnet14_cifar,
	'20': resnet20_cifar,
	'26': resnet26_cifar,
	'32': resnet32_cifar,
	'44': resnet44_cifar,
	'56': resnet56_cifar,
	'110': resnet110_cifar,
}