import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


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

class FocalLoss_with_mask(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss_with_mask, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets,mask):
        inputs = inputs*mask
        targets = targets*mask
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



class SpatialConsistencyLoss(nn.Module):
    def __init__(self, mu=0.34, std=0.04, patch_size=32):
        """
        Initializes the loss with Gaussian distribution parameters.
        :param mu: Mean of the Gaussian distribution.
        :param std: Standard deviation of the Gaussian distribution.
        :param patch_size: Size of each patch.
        """
        super(SpatialConsistencyLoss, self).__init__()
        self.mu = mu
        self.std = std
        self.patch_size = patch_size
        self.gaussian = norm(loc=mu, scale=std)

    def forward(self, unet_output, transformer_output):
        """
        Enforces patches in unet_output to align with Gaussian distribution or zero based on transformer_output.
        :param unet_output: Output from UNet branch.
        :param transformer_output: Output from Transformer branch.
        :return: Spatial consistency loss.
        """
        positive_loss = 0.0
        positive_cnt = 0
        negative_loss = 0.0
        negative_cnt = 0
        for i in range(0, unet_output.shape[2], self.patch_size):
            for j in range(0, unet_output.shape[3], self.patch_size):
                unet_patch = unet_output[:, :, i:i+self.patch_size, j:j+self.patch_size]
                ti = int(i/self.patch_size)
                tj = int(j/self.patch_size)
                transformer_value = transformer_output[:, ti, tj]
                patch_mean = unet_patch.mean()
                if transformer_value >= 0.5:  # Positive patch
                    # Probability of patch_mean under Gaussian distribution
                    prob = self.gaussian.pdf(patch_mean.item())
                    positive_loss += -torch.log(torch.tensor(prob + 1e-6))  # Add small value to avoid log(0)
                    positive_cnt +=1
                else:  # Negative patch
                    # Loss for negative patches to be close to zero
                    negative_loss += F.mse_loss(patch_mean, torch.tensor(0.0, device=patch_mean.device))
                    negative_cnt+=1

        positive_loss = positive_loss/positive_cnt
        negative_loss = negative_loss/negative_cnt

        loss = positive_loss + negative_loss

        return loss

