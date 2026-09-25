"""
mmfi_wifi.channel_trans — Channel Transformer (vendor 自 MetaFi)
=================================================================
来源: references/upstream/metafi/models/ChannelTrans.py
仅做清理, 数学逻辑不变。
"""

import copy
import logging
import math

import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm

logger = logging.getLogger(__name__)


class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, img_size, in_channels):
        super().__init__()
        n_patches = (img_size[0] * img_size[1])
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = 17, 12
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Attention_org(nn.Module):
    def __init__(self, vis, channel_num, num_heads):
        super().__init__()
        self.vis = vis
        self.KV_size = channel_num
        self.num_attention_heads = num_heads

        self.query1 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(num_heads):
            self.query1.append(copy.deepcopy(nn.Linear(channel_num, channel_num, bias=False)))
            self.key.append(copy.deepcopy(nn.Linear(self.KV_size, self.KV_size, bias=False)))
            self.value.append(copy.deepcopy(nn.Linear(self.KV_size, self.KV_size, bias=False)))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num, channel_num, bias=False)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

    def forward(self, emb1):
        multi_head_Q1 = torch.stack([q(emb1) for q in self.query1], dim=1)
        multi_head_K = torch.stack([k(emb1) for k in self.key], dim=1)
        multi_head_V = torch.stack([v(emb1) for v in self.value], dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K)
        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)
        attention_probs1 = self.softmax(self.psi(attention_scores1))

        weights = [attention_probs1.mean(1)] if self.vis else None

        attention_probs1 = self.attn_dropout(attention_probs1)

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V)
        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()
        context_layer1 = context_layer1.mean(dim=3)

        O1 = self.out1(context_layer1)
        O1 = self.proj_dropout(O1)
        return O1, weights


class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block_ViT(nn.Module):
    def __init__(self, vis, channel_num, num_heads):
        super().__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num, eps=1e-6)
        self.channel_attn = Attention_org(vis, channel_num, num_heads)
        self.ffn_norm1 = LayerNorm(channel_num, eps=1e-6)
        self.ffn1 = Mlp(channel_num, channel_num * expand_ratio)

    def forward(self, emb1):
        org1 = emb1
        cx1 = self.attn_norm1(emb1)
        cx1, weights = self.channel_attn(cx1)
        cx1 = org1 + cx1

        org1 = cx1
        x1 = self.ffn_norm1(cx1)
        x1 = self.ffn1(x1)
        x1 = x1 + org1
        return x1, weights


class Encoder(nn.Module):
    def __init__(self, vis, channel_num, num_layers, num_heads):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num, eps=1e-6)
        for _ in range(num_layers):
            self.layer.append(copy.deepcopy(Block_ViT(vis, channel_num, num_heads)))

    def forward(self, emb1):
        attn_weights = []
        for layer_block in self.layer:
            emb1, weights = layer_block(emb1)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1)
        return emb1, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, vis, img_size, channel_num, num_layers, num_heads):
        super().__init__()
        self.embeddings_1 = Channel_Embeddings(img_size=img_size, in_channels=channel_num)
        self.encoder = Encoder(vis, channel_num, num_layers, num_heads)
        self.reconstruct_1 = Reconstruct(channel_num, channel_num, kernel_size=1, scale_factor=(1, 1))

    def forward(self, en1):
        emb1 = self.embeddings_1(en1)
        encoded1, attn_weights = self.encoder(emb1)
        x1 = self.reconstruct_1(encoded1)
        x1 = x1 + en1
        return x1, attn_weights
