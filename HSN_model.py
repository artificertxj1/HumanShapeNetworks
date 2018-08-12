
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.init as init

class HSnet(nn.Module):
    """
    human shape net
    """
    def __init__(self, num_classes=20):
        super(HSnet, self).__init__()
        self.block1=nn.Sequential(
                nn.Conv2d(2,48,kernel_size=1,stride=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
        )
        self.block2=nn.Sequential(
                nn.Conv2d(48,128,kernel_size=5,stride=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
        )
        self.block3=nn.Sequential(
                nn.Conv2d(128,192,kernel_size=3, stride=3,padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
        )
        self.block4=nn.Sequential(
                nn.Conv2d(192,192,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
        )
        self.block5=nn.Sequential(
                nn.Conv2d(192,128,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        )
        self.fc=nn.Sequential(
                nn.Dropout(),
                nn.Linear(128*22*30, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048,num_classes),
        )
    
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(out.size(0), 128*22*30)
        out = self.fc(out)
        return out
        