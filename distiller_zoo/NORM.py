from __future__ import print_function

import torch.nn as nn


class NORMConnector(nn.Module):
    """Channel-expansion connector for NORM distillation (CVPR 2023).

    Replaces the in-model layer31 used in the original NORM student models.
    Takes the student's penultimate spatial feature and expands its channels
    to match the teacher's channel-repeated feature used in the MSE loss:

        f_s_expanded = connector(feat_s[-2])          # [B, t_ch * co_sponge, H, W]
        f_t_repeated = feat_t[-2].repeat(1, co_sponge, 1, 1)  # [B, t_ch * co_sponge, H, W]
        loss_norm = MSE(f_s_expanded, f_t_repeated) * co_sponge

    Args:
        s_channels:  Number of channels in student's penultimate feature.
        t_channels:  Number of channels in teacher's penultimate feature.
        co_sponge:   Channel expansion factor (default 4).
    """

    def __init__(self, s_channels, t_channels, co_sponge=4):
        super(NORMConnector, self).__init__()
        out_ch = t_channels * co_sponge
        self.expand = nn.Sequential(
            nn.Conv2d(s_channels, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.expand(x)
