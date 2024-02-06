# zloss
import torch.nn.functional as F
from torch import nn


class CrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
        :param predict: [batch, num_class, height, width]
        :param target: [batch, height, width]
        :return: entropy loss
        """
        target_mask = target != self.ignore_label  # [batch, height, width]筛选出所有需要训练的像素点标签
        target = target[target_mask]  # [num_pixels]
        batch, num_class, height, width = predict.size()
        predict = predict.permute(0, 2, 3, 1)  # [batch, height, width, num_class]
        predict = predict[target_mask.unsqueeze(-1).repeat(1, 1, 1, num_class)].view(-1, num_class)
        loss = F.cross_entropy(predict, target)
        return loss