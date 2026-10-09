from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StructuredTokenizer(nn.Module):
    """把 (B,3,114,10) 变成时间-天线 token。"""

    def __init__(self, n_subcarriers: int = 114, n_time: int = 10, n_links: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.n_time = n_time
        self.n_links = n_links
        self.n_tokens = n_time * n_links
        self.proj = nn.Linear(n_subcarriers, embed_dim)
        self.time_embed = nn.Parameter(torch.zeros(1, n_time, 1, embed_dim))
        self.link_embed = nn.Parameter(torch.zeros(1, 1, n_links, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.link_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # 原始输入的三个维度为：天线、子载波、时间。
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0], self.n_tokens, self.n_subcarriers)
        x = self.proj(x)
        pos = (self.time_embed + self.link_embed).reshape(1, self.n_tokens, -1)
        return x + pos


class StructuredViTEncoder(nn.Module):
    """保留时间轴和天线轴语义的 ViT 编码器。"""

    def __init__(self, embed_dim: int = 768, depth: int = 6, num_heads: int = 6,
                 ffn_dim: int = 3072, n_time: int = 10, n_links: int = 3):
        super().__init__()
        self.tokenizer = StructuredTokenizer(n_time=n_time, n_links=n_links, embed_dim=embed_dim)
        self.n_tokens = self.tokenizer.n_tokens
        self.out_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-LayerNorm is incompatible with nested-tensor optimization.
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _encode(self, tokens: Tensor) -> Tensor:
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return self.norm(self.blocks(torch.cat([cls, tokens], dim=1)))

    def forward(self, x: Tensor, visible_mask: Tensor | None = None,
                return_tokens: bool = False):
        tokens = self.tokenizer(x)
        if visible_mask is not None:
            ids = visible_mask.nonzero(as_tuple=False).reshape(x.shape[0], -1, 2)[:, :, 1]
            gather = ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, 1, gather)
        output = self._encode(tokens)
        return output if return_tokens else output[:, 0]

    def forward_visible(self, x: Tensor, visible_mask: Tensor):
        output = self.forward(x, visible_mask=visible_mask, return_tokens=True)
        ids = visible_mask.nonzero(as_tuple=False).reshape(x.shape[0], -1, 2)[:, :, 1]
        return output[:, 0], output[:, 1:], ids


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 2, num_heads: int = 6,
                 n_tokens: int = 30):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-LayerNorm is incompatible with nested-tensor optimization.
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, cls: Tensor, visible: Tensor, visible_ids: Tensor) -> Tensor:
        batch, n_tokens, dim = visible.shape[0], self.pos_embed.shape[1] - 1, visible.shape[-1]
        full = self.mask_token.expand(batch, n_tokens, dim).clone()
        full.scatter_(1, visible_ids.unsqueeze(-1).expand(-1, -1, dim), visible)
        sequence = torch.cat([cls[:, None], full], dim=1) + self.pos_embed
        return self.norm(self.blocks(sequence))[:, 1:]


class StructuredPoseModel(nn.Module):
    """结构化 ViT + 现有 17 关节回归头。"""

    def __init__(self, dropout_p: float = 0.3, **encoder_kwargs):
        super().__init__()
        from pose_ssl.model import PoseHead

        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        self.head = PoseHead(self.encoder.out_dim, dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        return self.head(self.encoder(x))


class WiFiJEPA(nn.Module):
    """MM-Fi 三路幅度 CSI 的 link-level latent prediction。"""

    def __init__(self, embed_dim: int = 768, depth: int = 6, num_heads: int = 6,
                 ffn_dim: int = 3072, predictor_depth: int = 2,
                 n_time: int = 10, n_links: int = 3, n_masked_links: int = 1):
        super().__init__()
        self.context_encoder = StructuredViTEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, ffn_dim=ffn_dim,
            n_time=n_time, n_links=n_links,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.target_encoder.eval()
        self.predictor = Predictor(
            embed_dim=embed_dim, depth=predictor_depth, num_heads=num_heads,
            n_tokens=n_time * n_links,
        )
        self.n_time = n_time
        self.n_links = n_links
        self.n_masked_links = n_masked_links

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(context.data, alpha=1.0 - momentum)

    def forward(self, x: Tensor, target_mask: Tensor | None = None):
        if target_mask is None:
            from .masking import make_link_mask
            target_mask = make_link_mask(
                x.shape[0], self.n_time, self.n_links, self.n_masked_links, x.device
            )
        visible_mask = ~target_mask
        cls, visible, visible_ids = self.context_encoder.forward_visible(x, visible_mask)
        predicted = self.predictor(cls, visible, visible_ids)
        with torch.no_grad():
            target = self.target_encoder(x, return_tokens=True)[:, 1:]
            target = F.layer_norm(target, (target.shape[-1],))
        loss = F.smooth_l1_loss(predicted[target_mask], target[target_mask])
        return loss, {
            "loss": float(loss.detach()),
            "masked_tokens": int(target_mask.sum()),
            "masked_links": self.n_masked_links,
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}
