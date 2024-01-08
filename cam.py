from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(121, 11, 1)#这俩卷积就是meta-learner
        self.conv2 = nn.Conv2d(11, 121, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):#[4, 5, 30, 36, 36]
        input_a = a

        a = a.mean(3)#[4, 5, 30, 36]
        a = a.transpose(1, 3) #把通道36 放到第二位
        a = F.relu(self.conv1(a))#c:36-6-36
        a = self.conv2(a) #
        a = a.transpose(1, 3)
        a = a.unsqueeze(3) 
        
        a = torch.mean(input_a * a, -1) 
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a 

    def forward(self, f1, f2):# [5, 640, 11, 11] [75, 640, 11, 11]
        n1, c, h, w = f1.size()
        n2 = f2.size(0)
        b=1
        f1 = f1.view(b, n1, c, -1) #[1, 5, 640, 121]
        f2 = f2.view(b, n2, c, -1) #[1,75, 640, 121]

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)
        # cross-attetnion
        a1 = torch.matmul(f1_norm, f2_norm) #矩阵相乘 [1, 5, 75, 121, 121] correlation
        a2 = a1.transpose(3, 4) #[1, 5, 75, 121, 121]
        a1 = self.get_attention(a1) #[1, 5, 75, 121] fusion
        a2 = self.get_attention(a2) #[1, 5, 75, 121]

        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)#[1, 5, 640, 121]*[1, 5, 75, 121] (a1就是map权重
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.view(b, n1, n2, c, h, w)

        return f1.transpose(1, 2), f2.transpose(1, 2) #[1, 75, 5, 640, 11, 11] [1, 75, 5, 640, 11, 11] 该样本针对每类的特征，每类对应该样本的特征
