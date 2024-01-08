from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):#score_map:【120,5,6,6】【120】 ----- [75, 5, 11, 11]
        input_ = inputs
        input_ = input_.view(input_.size(0), input_.size(1), -1)  # [60, 64, 121]  [150, 5, 169] --------[75, 5, 121]
        input_ = torch.clamp(input_, min=1e-5, max=1.0 - 1e-5)  # 添加了截断区间

        log_probs = self.logsoftmax(input_)# [120, 64, 36] [120, 5, 36]
        targets = torch.zeros(input_.size(0), input_.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        #targets = torch.ones(input_.size(0), input_.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 0)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()# [120, 64, 1] [120, 5, 1]
        a1 = - targets * log_probs # 这是在做最后一步 y*log(...)
        # print("a1_size:",a1.size())#[75, 5, 121]
        # print("a1:",a1)
        loss = (- targets * log_probs).mean(0).sum() # 就是 一个batch里的所有q 找出数来代表对应这类的得分
        return loss / input_.size(2)#相当于最后一维求均值
