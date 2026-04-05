from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTM(nn.Module):
    """Temperature Transform Module, logit-based KD loss.

    Raises teacher softmax probabilities to power l, renormalizes,
    then computes KL divergence with student log-probabilities.

    Args:
        l: exponent applied to teacher softmax (default 1 = standard softmax)
    """
    def __init__(self, l):
        super(TTM, self).__init__()
        self.l = l

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)
        KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
        loss = torch.mean(KL)
        return loss


class WTTM(nn.Module):
    """Weighted Temperature Transform Module, logit-based KD loss.

    Same as TTM but weights the KL loss by the normalization constant,
    giving higher weight to samples where the temperature transform is
    more significant.

    Args:
        l: exponent applied to teacher softmax (default 1 = standard softmax)
    """
    def __init__(self, l):
        super(WTTM, self).__init__()
        self.l = l

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)
        KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
        loss = torch.mean(norm * KL)
        return loss
