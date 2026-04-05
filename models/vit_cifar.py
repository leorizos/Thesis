from __future__ import print_function

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class CifarViT(nn.Module):
    """Wraps timm VisionTransformer for CIFAR-100 (32x32, patch_size=4).

    is_feat interface:
        forward(x)              -> logits          [B, num_classes]
        forward(x, is_feat=True) -> (feats, logits)
            feats: list of 12 CLS tokens (x[:,0]) after each transformer block
                   each tensor shape [B, embed_dim]
    """

    def __init__(self, embed_dim, depth, num_heads, drop_path_rate=0.0, num_classes=100):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x, is_feat=False):
        v = self.vit

        # Patch embedding
        x = v.patch_embed(x)                           # [B, N, embed_dim]

        # Prepend CLS token and add positional embedding
        cls = v.cls_token.expand(x.shape[0], -1, -1)   # [B, 1, embed_dim]
        x = torch.cat((cls, x), dim=1)                 # [B, N+1, embed_dim]
        x = v.pos_drop(x + v.pos_embed)

        # Walk through transformer blocks, collecting CLS token after each
        feats = []
        for blk in v.blocks:
            x = blk(x)
            feats.append(x[:, 0])                      # CLS token [B, embed_dim]

        x = v.norm(x)
        logits = v.head(x[:, 0])                       # [B, num_classes]

        if is_feat:
            return feats, logits
        return logits


def vit_small_cifar(**kwargs):
    return CifarViT(embed_dim=384, depth=12, num_heads=6, **kwargs)


def vit_base_cifar(**kwargs):
    return CifarViT(embed_dim=768, depth=12, num_heads=12, **kwargs)
