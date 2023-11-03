import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self, positive_weight=5.0):
        super(WeightedMSELoss, self).__init__()
        self.positive_weight = positive_weight

    def forward(self, inputs, targets):
        weights = torch.ones_like(targets) * (targets == 1) * (self.positive_weight - 1) + 1
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        weighted_mse_loss = mse_loss * weights
        return weighted_mse_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        targets = targets.type(inputs.type())
        pt = (1 - targets) * (1 - inputs) + targets * inputs
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, positive_weight=1.0, dice_weight=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.weighted_mse_loss = WeightedMSELoss(positive_weight)
        self.dice_loss = DiceLoss(smooth)
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        w_mse = self.weighted_mse_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        combined_loss = (1 - self.dice_weight) * w_mse + self.dice_weight * dice
        return combined_loss



