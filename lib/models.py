# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from collections import OrderedDict
from torch.nn import init


# Ensemble Model
class Ensemble(nn.Module):
    def __init__(self, source_list, detach_list):
        super(Ensemble, self).__init__()
        self.source_list = source_list
        self.detach_list = detach_list
        self.dummy_module = torch.nn.Linear(1,1, bias=False)
    
    def forward(self, x):
        return None
    
    def post_forward(self, outputs):
        # pick up ensemble models
        filtered_outputs = []
        for id_ in self.source_list:
            if id_ in self.detach_list:
                # detach gragh
                filtered_outputs += [outputs[id_][0].detach()]
            else:
                filtered_outputs += [outputs[id_][0]] # [[100]] += [100] --> [ [100], [100] ]
                
        # compute ensemble output
        ensembled_output = torch.stack(filtered_outputs).sum(dim=0) # [ [100], [100], [100] ] -> stack [3*100] -> sum [100]
        
        return [ensembled_output, None, None]


# Attention branch network (ABN) based on ResNet
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


# ABN ResNet -------------------------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        self.att_layer4 = self._make_layer(block, 512, layers[3], stride=1, down_size=False)
        self.bn_att = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(14)
        self.sigmoid = nn.Sigmoid()

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True,):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ax = self.bn_att(self.att_layer4(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        bs, cs, ys, xs = ax.shape
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))

        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * self.att
        #rx = rx + x   # w/o residual functions
        per = rx
        rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)

        return [rx, ax, self.att]


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def load_weight_ABN(model, block, num_classes, weight_path):
    print("load_weight : ABN")
    state_dict = torch.load(weight_path)['state_dict']
    model.load_state_dict(fix_model_state_dict(state_dict))
    
    # change number of classes in output layer
    model.att_conv  = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
    model.bn_att2   = nn.BatchNorm2d(num_classes)
    model.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
    model.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    return model

def load_weight_ResNet(model, block, num_classes, weight_path):
    print("load_weight : ResNet")
    model.load_state_dict(model_zoo.load_url(weight_path), strict=False)
    
    # change number of classes in output layer      
    model.att_conv  = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
    model.bn_att2   = nn.BatchNorm2d(num_classes)
    model.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
    model.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    return model


def resnet18_abn(num_classes=1000, pre_ABN=False, pre_ResNet=False):
    print("network architecture : ABN based on ResNet18")
    if pre_ABN:
        model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
        weight_path = 'pre-train/ImageNet/resnet18/model_best.pth.tar'
        model = load_weight_ABN(model, BasicBlock, num_classes, weight_path)
    elif pre_ResNet:
        model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
        weight_path = model_urls['resnet18']
        model = load_weight_ResNet(model, BasicBlock, num_classes, weight_path)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34_abn(num_classes=1000, pre_ABN=False, pre_ResNet=False):
    print("network architecture : ABN based on ResNet34")
    if pre_ABN:
        model = ResNet(BasicBlock, [3, 4, 6, 3], 1000)
        weight_path = 'pre-train/ImageNet/resnet34/model_best.pth.tar'
        model = load_weight_ABN(model, BasicBlock, num_classes, weight_path)
    elif pre_ResNet:
        model = ResNet(BasicBlock, [3, 4, 6, 3], 1000)
        weight_path = model_urls['resnet34']
        model = load_weight_ResNet(model, BasicBlock, num_classes, weight_path)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model


def resnet50_abn(num_classes=1000, pre_ABN=False, pre_ResNet=False):
    print("network architecture : ABN based on ResNet50")
    if pre_ABN:
        model = ResNet(Bottleneck, [3, 4, 6, 3], 1000)
        weight_path = 'pre-train/ImageNet/resnet50/model_best.pth.tar'
        model = load_weight_ABN(model, Bottleneck, num_classes, weight_path)
    elif pre_ResNet:
        model = ResNet(Bottleneck, [3, 4, 6, 3], 1000)
        weight_path = model_urls['resnet50']
        model = load_weight_ResNet(model, Bottleneck, num_classes, weight_path)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def resnet101_abn(num_classes=1000, pre_ABN=False, pre_ResNet=False):
    print("network architecture : ABN based on ResNet101")
    if pre_ABN:
        model = ResNet(Bottleneck, [3, 4, 23, 3], 1000)
        weight_path = 'pre-train/ImageNet/resnet101/model_best.pth.tar'
        model = load_weight_ABN(model, Bottleneck, num_classes, weight_path)
    elif pre_ResNet:
        model = ResNet(Bottleneck, [3, 4, 23, 3], 1000)
        weight_path = model_urls['resnet101']
        model = load_weight_ResNet(model, Bottleneck, num_classes, weight_path)
    else:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def resnet152_abn(num_classes=1000, pre_ABN=False, pre_ResNet=False):
    print("network architecture : ABN based on ResNet152")
    if pre_ABN:
        model = ResNet(Bottleneck, [3, 8, 36, 3], 1000)
        weight_path = 'pre-train/ImageNet/resnet152/model_best.pth.tar'
        model = load_weight_ABN(model, Bottleneck, num_classes, weight_path)
    elif pre_ResNet:
        model = ResNet(Bottleneck, [3, 8, 36, 3], 1000)
        weight_path = model_urls['resnet152']
        model = load_weight_ResNet(model, Bottleneck, num_classes, weight_path)
    else:
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return model


# Vanilla ResNet -------------------------------------------------------------------------------
class ResNet_vanilla(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_vanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        
        x_out = self.avgpool(x_4)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc(x_out)

        return [x_out, None, x_4.pow(2).mean(1)]
    
    
def resnet18_vanilla(num_classes=1000):
    print("network architecture : ResNet18")
    model = ResNet_vanilla(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


# CIFAR ABN -------------------------------------------------------------------------------
class CIFAR_ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(CIFAR_ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, down_size=True)
        self.layer2 = self._make_layer(block, 32, n, stride=2, down_size=True)

        self.att_layer3 = self._make_layer(block, 64, n, stride=1, down_size=False)
        self.bn_att = nn.BatchNorm2d(64 * block.expansion)
        self.att_conv = nn.Conv2d(64 * block.expansion, num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(16)
        self.sigmoid = nn.Sigmoid()

        self.layer3 = self._make_layer(block, 64, n, stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        ax = self.bn_att(self.att_layer3(x))
        ax = self.relu(self.bn_att2(self.att_conv(ax)))
        bs, cs, ys, xs = ax.shape
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))
        
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)

        rx = x * self.att
        rx = rx + x
        rx = self.layer3(rx)
        rx = self.avgpool(rx)
        rx = rx.view(rx.size(0), -1)
        rx = self.fc(rx)

        return [rx, ax, self.att]


def resnet20_abn(num_classes=10):
    print("network architecture : ABN based on ResNet20")
    model = CIFAR_ResNet(20, num_classes)
    return model


# Vanilla CIFAR ResNet -------------------------------------------------------------------------------
class DownsampleA(nn.Module):  

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__() 
        assert stride == 2    
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

    def forward(self, x):   
        x = self.avg(x)  
        return torch.cat((x, x.mul(0)), 1)  

class DownsampleC(nn.Module):     

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)

class CIFAR_ResNet_vanilla(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes):
        """ Constructor
        Args:
            depth: number of layers.
            num_classes: number of classes
            base_width: base width
        """
        super(CIFAR_ResNet_vanilla, self).__init__()
        
        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        #print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)        
        self.classifier = nn.Linear(64*block.expansion, num_classes) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
   
    
def resnet20(num_classes=10):
    print("network architecture : ResNet20")
    model = CIFAR_ResNet_vanilla(ResNetBasicblock, 20, num_classes)
    return model