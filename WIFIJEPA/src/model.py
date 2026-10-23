from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .paths import MAE_ENCODER_CONTRACT


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
        # pre-LN ? nested tensor ???????????????????
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

    def encode_visible(self, x: Tensor, ids_keep: Tensor) -> Tensor:
        """Encode visible tokens without a CLS token, matching ViT-MAE."""
        tokens = self.tokenizer(x)
        gather = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return self.norm(self.blocks(torch.gather(tokens, 1, gather)))




class StructuredViTMAE(nn.Module):
    """MAE wrapper using the same StructuredViT encoder as fine-tuning.

    The masking, decoder token restoration, per-token normalization, and
    masked-only reconstruction loss follow the original ViT-MAE contract.
    StructuredViT tokens are one (time, link) location with all subcarriers.
    """

    encoder_contract = MAE_ENCODER_CONTRACT

    def __init__(self, embed_dim=768, depth=6, num_heads=6, ffn_dim=3072,
                 n_time=10, n_links=3, mask_ratio=0.75, decoder_dim=512,
                 decoder_depth=8, decoder_heads=16, n_subcarriers=114):
        super().__init__()
        if not 0.0 < float(mask_ratio) < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        if decoder_dim % decoder_heads:
            raise ValueError("decoder_dim must be divisible by decoder_heads")
        self.encoder = StructuredViTEncoder(embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, ffn_dim=ffn_dim, n_time=n_time, n_links=n_links)
        if self.encoder.tokenizer.n_subcarriers != n_subcarriers:
            raise ValueError("n_subcarriers must match StructuredTokenizer")
        self.mask_ratio = float(mask_ratio)
        self.n_subcarriers = int(n_subcarriers)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_cls_pos = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.encoder.n_tokens, decoder_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            layer, num_layers=decoder_depth, enable_nested_tensor=False
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.n_subcarriers)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_cls_pos, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def forward(self, x: Tensor):
        if x.ndim == 5:
            x = x.squeeze(1)
        expected = (self.encoder.tokenizer.n_links, self.n_subcarriers,
                    self.encoder.tokenizer.n_time)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected CSI shape (B,{expected[0]},{expected[1]},{expected[2]})")

        # StructuredTokenizer's token order is (time, link); each target is
        # the original subcarrier vector at one structured token location.
        targets = x.permute(0, 3, 1, 2).reshape(
            x.shape[0], self.encoder.n_tokens, self.n_subcarriers
        )
        mean = targets.mean(dim=-1, keepdim=True)
        var = targets.var(dim=-1, keepdim=True, unbiased=True)
        targets = (targets - mean) / (var + 1e-6).sqrt()

        batch, n_tokens = x.shape[0], self.encoder.n_tokens
        n_keep = max(1, int(n_tokens * (1.0 - self.mask_ratio)))
        ids_shuffle = torch.argsort(torch.rand(batch, n_tokens, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        tokens = self.encoder.tokenizer(x)
        gather = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        visible = torch.gather(tokens, 1, gather)
        # Match the downstream encoding path so CLS receives reconstruction gradients.
        encoded = self.decoder_embed(self.encoder._encode(visible))
        mask_tokens = self.mask_token.expand(batch, n_tokens - n_keep, -1)
        patches = torch.cat([encoded[:, 1:], mask_tokens], dim=1)
        restore = ids_restore.unsqueeze(-1).expand(batch, n_tokens, patches.shape[-1])
        patches = torch.gather(patches, 1, restore) + self.decoder_pos
        decoded = torch.cat([encoded[:, :1] + self.decoder_cls_pos, patches], dim=1)
        predicted = self.decoder_pred(
            self.decoder_norm(self.decoder_blocks(decoded))
        )[:, 1:]

        loss = (predicted - targets).pow(2).mean(dim=-1)
        mask = torch.ones(batch, n_tokens, device=x.device)
        mask[:, :n_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        loss = (loss * mask).sum() / mask.sum()
        return loss, {
            "loss": float(loss.detach()),
            "mask_ratio": float(mask.mean().detach()),
            "n_keep": n_keep,
            "masked_tokens": int(mask.sum()),
            "encoder_contract": self.encoder_contract,
        }

    def export_encoder_state_dict(self):
        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}


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
        # pre-LN ? nested tensor ???????????????????
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


class JointQueryPoseModel(nn.Module):
    """Structured CSI encoder with one continuous query per output joint."""

    def __init__(self, dropout_p: float = 0.3, n_joints: int = 17,
                 query_depth: int = 2, query_heads: int | None = None,
                 **encoder_kwargs):
        super().__init__()
        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        dim = self.encoder.out_dim
        query_heads = query_heads or encoder_kwargs.get("num_heads", 6)
        if dim % query_heads:
            raise ValueError("query_heads must divide encoder embedding dimension")
        self.query_embed = nn.Parameter(torch.zeros(1, n_joints, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=query_heads,
            dim_feedforward=dim * 4,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(layer, num_layers=query_depth)
        self.query_norm = nn.LayerNorm(dim)
        self.pose = nn.Linear(dim, 3)
        nn.init.trunc_normal_(self.query_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        memory = self.encoder(x, return_tokens=True)
        queries = self.query_embed.expand(x.shape[0], -1, -1)
        decoded = self.query_norm(self.query_decoder(queries, memory))
        return self.pose(decoded)


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

    def forward(self, x: Tensor, target_mask: Tensor | None = None,
                return_context: bool = False):
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
        log = {
            "loss": float(loss.detach()),
            "masked_tokens": int(target_mask.sum()),
            "masked_links": self.n_masked_links,
        }
        return (loss, log, cls) if return_context else (loss, log)

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}
