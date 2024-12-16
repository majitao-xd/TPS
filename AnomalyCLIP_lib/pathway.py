# Copyright (c) 2024 Jitao Ma

import torch
import math
import torch.nn.functional as F
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PathWay(torch.nn.Module):
    def __init__(self, text_projection, img_projection):
        super(PathWay, self).__init__()
        self.text_projection = text_projection
        self.img_projection = img_projection
        transformer_width = text_projection.data.shape[-1]

        drop = 0.

        self.ln_0 = nn.LayerNorm(transformer_width)
        self.ln_1 = nn.LayerNorm(transformer_width)
        self.text_attention = nn.MultiheadAttention(transformer_width, 8, batch_first=True, dropout=drop)
        self.ln_2 = nn.LayerNorm(transformer_width)
        self.cross_attention = nn.MultiheadAttention(transformer_width, 8, batch_first=True, dropout=drop)
        self.ln_3 = nn.LayerNorm(transformer_width)
        self.text_feed = nn.Sequential(
            nn.Linear(transformer_width, transformer_width * 4),
            QuickGELU(),
            nn.Linear(transformer_width * 4, transformer_width),)

        self.ln_4 = nn.LayerNorm(transformer_width)
        self.img_attention = nn.MultiheadAttention(transformer_width, 8, batch_first=True, dropout=drop)
        self.ln_5 = nn.LayerNorm(transformer_width)
        self.img_feed = nn.Sequential(
            nn.Linear(transformer_width, transformer_width * 4),
            QuickGELU(),
            nn.Linear(transformer_width * 4, transformer_width),)


    def forward(self, text_features_ori, class_features_ori, patch_features_ori, tokenized_prompts=None):
        class_features = self.ln_0(class_features_ori)
        text_features = self.ln_1(text_features_ori)
        patch_features = self.ln_4(patch_features_ori)

        class_features = class_features + self.text_attention(class_features, patch_features, patch_features)[0]
        class_features = self.ln_2(class_features)
        class_features = class_features + self.cross_attention(class_features, text_features, class_features)[0]
        class_features = class_features + self.text_feed(self.ln_3(class_features))

        return class_features, patch_features

