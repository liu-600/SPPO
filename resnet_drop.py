import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            #print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            #print (block_mask.size())
            #print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class LayerNorm(nn.LayerNorm):
    """ Layernorm f or channels of '2d' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class ResNet(nn.Module):

    def __init__(self, num_classes,drop_block=False, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        self.nFeat = 640
        super(ResNet, self).__init__()
        self.sematic_channel = False
        self.prompt2 = 0
        expansion = 1

        ####
        self.drop_rate = drop_rate  # 几乎固定
        self.block_size = dropblock_size  # 几乎固定
        self.stride = 1  # 大部分情况赋2

        self.register_buffer('num_batches_tracked', torch.tensor(0))  # 会一直增加
        self.drop_block = drop_block  # FFTT

        self.pool = True  #
        # head

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        embed_dim = 640
        self.norm = nn.BatchNorm2d(embed_dim, eps=1e-5, momentum=0.1, track_running_stats=True)
        self.head = nn.Linear(embed_dim, num_classes)


        #stage 1
        self.inplanes = 3
        planes = 64
        stride = 2
        self.stage1 = nn.Sequential(
            conv3x3(self.inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.1),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),)
        self.maxpool1 = nn.MaxPool2d(stride, ceil_mode=True)
        self.DropBlock1 = DropBlock(block_size=self.block_size)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion),
        )
        #stage 2
        self.inplanes = 64
        planes = 160
        stride = 2
        self.stage2 = nn.Sequential(
            conv3x3(self.inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.1),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes), )
        self.maxpool2 = nn.MaxPool2d(stride, ceil_mode=True)
        self.DropBlock2 = DropBlock(block_size=self.block_size)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes *expansion),
        )
        #stage 3
        self.inplanes = 160
        planes = 320
        stride = 2
        self.stage3 = nn.Sequential(
            conv3x3(self.inplanes, planes),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.1),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes), )
        self.maxpool3 = nn.MaxPool2d(stride, ceil_mode=True)
        self.DropBlock3 = DropBlock(block_size=self.block_size)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion),
        )
        #stage 4
        self.inplanes = 320
        planes = 640
        stride = 2
        self.conv1 = conv3x3(self.inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool4 = nn.MaxPool2d(stride, ceil_mode=True)
        self.DropBlock4 = DropBlock(block_size=self.block_size)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * expansion),
        )
        ####
        # 只是对downsample里的参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False, return_both=False, return_map=False):#Q 走这
        #print("x0:",x.size())#[70, 3, 84, 84]
        self.num_batches_tracked += 1
    #stage 1
        self.drop_block = False
        self.pool = True
        residual = x
        out = self.stage1(x)
        if self.downsample1 is not None:
            residual = self.downsample1(x)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool1(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock1(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x1=out
        #print("x1:", x1.size())
    #satge 2
        self.drop_block = False
        self.pool = True
        residual = x1
        out = self.stage2(x1)
        if self.downsample2 is not None:
            residual = self.downsample2(x1)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool2(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock2(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x2 = out
        #print("x2:", x2.size())
    #stage 3
        self.drop_block = True
        self.pool = True
        residual = x2
        out = self.stage3(x2)
        if self.downsample3 is not None:
            residual = self.downsample3(x2)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool3(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock3(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x3 = out
        #print("x3:", x3.size())
    #stage 4
        self.drop_block = True
        self.pool = False
        residual = x3
        out = self.conv1(x3)  # [70, 640, 11, 11]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # [70, 640, 11, 11]
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # [70, 640, 11, 11]
        out = self.bn3(out)

        if self.downsample4 is not None:
            residual = self.downsample4(x3)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool4(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock4(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x4 = out
        #print("x4:", x4.size())
        # head
        #print("x0:", x4.size())#[128, 640, 11, 11]
        x = self.norm(x4)
        #print("x1:", x.size())#[128, 640, 11, 11]
        x = self.global_pooling(x)

        logit = self.head(x.view(x.size(0), -1))


        return logit,x4


    def forward_with_semantic_prompt_channel(self, x, semantic_prompt, args):# S 走这

        if 'channel' in args.prompt_mode:  #
            self.prompt2 = self.t2i2(semantic_prompt)#[70,640]

        self.num_batches_tracked += 1
        # stage 1
        self.drop_block = False
        self.pool = True
        residual = x
        out = self.stage1(x)
        if self.downsample1 is not None:
            residual = self.downsample1(x)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool1(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock1(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x1 = out

        # satge 2
        self.drop_block = False
        self.pool = True
        residual = x1
        out = self.stage2(x1)
        if self.downsample2 is not None:
            residual = self.downsample2(x1)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool2(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock2(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x2 = out
        # stage 3
        self.drop_block = True
        self.pool = True
        residual = x2
        out = self.stage3(x2)
        if self.downsample3 is not None:
            residual = self.downsample3(x2)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool3(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock3(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x3 = out
        # stage 4
        self.drop_block = True
        self.pool = False
        residual = x3
        out = self.conv1(x3)  # [70, 640, 11, 11]
        out = self.bn1(out)
        out = self.relu(out)
        ##########
        B, C, H, W = out.size()
        # 开始加入
        #print("prompt2:",self.prompt2.size(),"out:",out.size())#[5, 640] [5, 640, 11, 11]
        context = out.view(B, C, -1).mean(-1)
        #print("context1:",context.size())#[5, 640]
        context = torch.cat([context, self.prompt2], dim=-1)
        #print("context2:", context.size())#[5, 1280]
        context = self.se_block(context)
        context = context - context.mean(dim=-1, keepdim=True)
        out = out + context.view(B, C, 1, 1)

        ##########
        out = self.conv2(out)  # [70, 640, 11, 11]
        out = self.bn2(out)
        out = self.relu(out)
        ##########
        B, C, H, W = out.size()
        # 开始加入
        # print("prompt2:",prompt2.size(),"context:",context.size())
        context = out.view(B, C, -1).mean(-1)#[70,640]
        # print("context:",context.size())
        context = torch.cat([context, self.prompt2], dim=-1)#[70,1280]
        context = self.se_block(context)#[70,640]
        context = context - context.mean(dim=-1, keepdim=True)
        out = out + context.view(B, C, 1, 1)
        ##########
        out = self.conv3(out)  # [70, 640, 11, 11]
        out = self.bn3(out)


        if self.downsample4 is not None:
            residual = self.downsample4(x3)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool4(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock4(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        x4 = out


        return None,x4
    def forward_as_dict(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #result = self.fc(x)
        return output

def resnet12(num_classes,drop_block=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet( num_classes,drop_block=drop_block, **kwargs)
    return model
