import torch
import torch.nn as nn
import kornia

from .balancer import Balancer
from pcdet.utils import depth_utils


class DDNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_weight,
                 bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = kornia.losses.FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            depth_maps: torch.Tensor(B, H, W)]: Depth map [m]
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        tb_dict = {}
        # import pdb; pdb.set_trace()
        # Bin depth map to create target
        depth_target = depth_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)

        # Final loss
        loss *= self.weight
        tb_dict.update({"ddn_loss": loss.item()})

        return loss, tb_dict
