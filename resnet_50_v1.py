import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision


class Resnet_50_v1(nn.Module):
    def __init__(self, pretrain=True):
        self.resnet = torchvision.models.resnet50(pretrained=pretrain)
        self.conv1 = self.resnet.conv1 
        self.bn1 = self.resnet.bn1 
        self.relu = self.resnet.relu 
        self.maxpool = self.resnet.maxpool 

        self.layer1 = self.resnet.layer1 
        self.layer2 = self.resnet.layer2 
        self.layer3 = self.resnet.layer3 
        self.layer4 = self.resnet.layer4 

        self.avgpool = self.resnet.avgpool 
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    