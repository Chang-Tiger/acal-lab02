#hw2-4
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
torch.manual_seed(0)
NUM_classes = 1000
BLOCK_SIZE = 64

class MyLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def get_split_size_(self, length: int, BLOCK_SIZE: int) -> List[int]:
        num = length // BLOCK_SIZE
        remainder = length % BLOCK_SIZE
        split_size = []
        for i in range(num):
            split_size.append(BLOCK_SIZE)
        if(remainder != 0):
            num += 1
            split_size.append(remainder)
        return split_size
    def forward(self, input: Tensor) -> Tensor:
        # print("Input shape:", input.shape)
        # if(input.size(0)!=1):
        #     input = torch.flatten(input, 1)
        M = input.size(0)
        N = self.in_features
        K = self.out_features
        split_size = self.get_split_size_(K, BLOCK_SIZE)
        a_split = torch.split(input, split_size_or_sections=BLOCK_SIZE, dim=1)
        b = self.weight
        bias = self.bias
        b_dim1 = torch.split(b, split_size_or_sections=BLOCK_SIZE, dim=1)
        b_blk = []
        for b_0 in b_dim1:
            b_dim0 = torch.split(b_0, split_size_or_sections=BLOCK_SIZE, dim=0)
            b_blk.append(b_dim0)
        
        c_blk = []
        for i in range(len(b_blk)):
            tmp_matrix = torch.zeros(M, split_size[i])
            for j in range(len(b_blk[i])):
                tmp_matrix += a_split[j] @ b_blk[i][j]
            c_blk.append(tmp_matrix)
        
        c = torch.cat(c_blk, dim=1)
        c += bias 
        return c

class modified_AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.maxpool1 =  nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(1,-1)
        )
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            MyLinear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MyLinear(4096, 4096),
            nn.ReLU(inplace=True),
            MyLinear(4096, 1000)
        )        
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x=self.conv1(x)
        x=self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x= self.conv5(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        x = self.classifier(x)        
        return x

modified_model = modified_AlexNet(num_classes=NUM_classes)
input_data = torch.randn(1, 3, 224, 224)  # Assuming batch size is 1
modified_model.eval()
traced_script_module = torch.jit.trace(modified_model, input_data)#, check_trace=False
traced_script_module.save("tracedmodified_alexnet_new.pt")
