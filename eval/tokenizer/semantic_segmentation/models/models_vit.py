# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for semantic seg."""
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        norm_layer = kwargs['norm_layer']
        patch_size = kwargs['patch_size']
        self.num_layers =  len(self.blocks) + 1

        self.fc_norm = norm_layer(embed_dim)
        del self.norm

        self.patch_embed = PatchEmbed(img_size=3,
                                      patch_size=patch_size,
                                      in_chans=3,
                                      embed_dim=embed_dim)
        assert self.num_classes > 0
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, 1)
        # manually initialize fc layer
        trunc_normal_(self.head.weight, std=2e-5)

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def forward_features(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 1:, :]
        x = self.fc_norm(x)
        b, _, c = x.shape
        ih, iw = w // self.patch_embed.patch_size, \
            h // self.patch_embed.patch_size
        x = x.view(b, ih, iw, c).permute(0, 3, 1, 2).contiguous()

        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid
        # floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2).contiguous(),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).contiguous().view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1)

    def get_layer_id(self, name):
        """Assign a parameter with its layer id Following BEiT: https://github.com/
        microsoft/unilm/blob/master/beit/optim_factory.py#L33.
        
        For each layer, the get_layer_id returns (layer_group, layer_id). 
        According to the layer_group, different parameters are grouped, 
        and layers in different groups use different decay rates.

        If only the layer_id is returned, the layer_group are set to 0 by default.
        """
        if name in ['cls_token', 'pos_embed']:
            return (0, 0)
        elif name.startswith('patch_embed'):
            return (0, 0)
        elif name.startswith('blocks'):
            return (0, int(name.split('.')[1]) + 1)
        else:
            return (0, self.num_layers)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        return x


def vit_small_patch16(args):
    kwargs = dict(
        num_classes=args.nb_classes, 
        drop_path_rate=getattr(args, 'drop_path', 0)
    )
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model


def vit_base_patch16(args):
    kwargs = dict(
        num_classes=args.nb_classes, 
        drop_path_rate=getattr(args, 'drop_path', 0)
    )
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model
