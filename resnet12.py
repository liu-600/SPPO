import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, kernel=3):
        self.inplanes = 64
        self.kernel = kernel
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.nFeat = 512 * block.expansion

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("x0:",x.size())#[220, 64, 84, 84]
        x = self.layer1(x)
        #print("x1:", x.size())#[220, 64, 42, 42]
        x = self.layer2(x)
        #print("x2:", x.size())#[220, 128, 21, 21]
        x = self.layer3(x)
        #print("x3:", x.size())#[220, 256, 11, 11]
        x = self.layer4(x)
        #print("x4:", x.size())#[220, 512, 6, 6]

        return x
    def forward_with_semantic_prompt_channel(self, x, semantic_prompt, args):
        if 'spatial' in args.prompt_mode:
            prompt1 = self.t2i(semantic_prompt)
        if 'channel' in args.prompt_mode:#
            prompt2 = self.t2i2(semantic_prompt)

        if self.using_stem:
            x = self.stem(x)

        # stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("x0:",x.size())#[220, 64, 84, 84]
        x = self.layer1(x)

        # stage 2
        if not self.vit_embedding:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = x + self.pos_embed2
                x = self.pos_drop(x)
        stage = 2.0
        #print("xxxxx:",x.size())#[5, 192, 14, 14]
        for b in self.stage2:
            if np.absolute(stage - args.stage) < 1e-6:
                B, C, H, W = x.shape
                if 'channel' in args.prompt_mode:#开始通道维度的融合
                    #print("prompt2:",prompt2.size(),"context:",context.size())
                    context = x.view(B, C, -1).mean(-1)
                    #print("context:",context.size())
                    context = torch.cat([context, prompt2], dim=-1)
                    context = self.se_block(context)
                    context = context - context.mean(dim=-1, keepdim=True)
                    x = x + context.view(B, C, 1, 1)
                if 'spatial' in args.prompt_mode:
                    prompt1 = prompt1.view(B, C, 1, 1).repeat(1, 1, 1, W)
                    x = torch.cat([x, prompt1], dim=2)
            x = b(x)
            stage += 0.1
        if 'spatial' in args.prompt_mode and 2 <= args.stage < 3:
            x = x[:, :, :H]

        # stage3
        if not self.vit_embedding:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = x + self.pos_embed3
                x = self.pos_drop(x)
        stage = 3.0
        for b in self.stage3:
            if np.absolute(stage - args.stage) < 1e-6:
                B, C, H, W = x.shape
                if 'channel' in args.prompt_mode:
                    #print("prompt2:", prompt2.size(), "x:", x.size())#[5, 384] [5, 384, 7, 7]
                    context = x.view(B, C, -1).mean(-1)#[5, 384] 将7*7的矩阵取了均值
                    context = torch.cat([context, prompt2], dim=-1)
                    context = self.se_block(context)
                    context = context - context.mean(dim=-1, keepdim=True)
                    x = x + context.view(B, C, 1, 1)#[5, 384, 7, 7]
                if 'spatial' in args.prompt_mode:
                    prompt1 = prompt1.view(B, C, 1, 1).repeat(1, 1, 1, W)#[5, 384, 1, 7]
                    x = torch.cat([x, prompt1], dim=2)#[5, 384, 8, 7]
            x = b(x)
            stage += 0.1

        # head
        x = self.norm(x)
        if self.pool:
            if 'spatial' not in args.prompt_mode or args.stage < 3:
                x = self.global_pooling(x)
            else:
                B, C, H, W = x.shape
                if args.avg == 'all':
                    x = x.view(B, C, -1)[:, :, :(H - 1) * W + 1].mean(-1)
                elif args.avg == 'patch':
                    x = x.view(B, C, -1)[:, :, :(H - 1) * W].mean(-1)
                elif args.avg == 'head':
                    x = x.view(B, C, -1)[:, :, -1]
        else:
            x = x[:, :, 0, 0]

        logit = self.head( x.view(x.size(0), -1) )
        return logit, x.squeeze()

def resnet12(num_classes=num_classes):
    model = ResNet(BasicBlock, [1,1,1,1], kernel=3)
    return model
