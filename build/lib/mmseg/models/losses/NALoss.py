import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal

from ..builder import LOSSES
from .utils import get_class_weight


@LOSSES.register_module()
class NALoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_weight=1.0, class_weight=None, thres: float = 0.7,
                 min_kept: int = 1000000, loss_name='AGDRLoss'):
        super(NALoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        # frequency_list training methods
        self.frequency_list = frequency_list.sum() - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = 255

    def forward(self, pred, target, weight, ignore_index=255, avg_factor=None, reduction_override=None):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        self.frequency_list = self.frequency_list.to(pred.device)
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.sum() * self.frequency_list).permute(
            0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, weight=class_weight, reduction='none', ignore_index=ignore_index)
        if weight is not None:
            weight = weight.float()
        x = loss.contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label  # (N*H*W)

        tmp_target = target.clone()  # (N, H, W)
        tmp_target[tmp_target == self.ignore_label] = 0
        # pred: (N, C, H, W) -> (N*H*W, C)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        # pred: (N*H*W, C) -> (N*H*W), ind: (N*H*W)
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        if pred.numel() > 0:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        else:
            return pred.new_tensor(0.0)
        threshold = max(min_value, self.thresh)

        pixel_losses = x[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        # loss = self.loss_weight*weight_reduce_loss(
        #     loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        loss = self.loss_weight * pixel_losses.mean()
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
