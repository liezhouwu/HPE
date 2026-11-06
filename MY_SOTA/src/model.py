from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _squeeze_csi(x: Tensor) -> Tensor:
    if x.ndim == 5 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 4:
        raise ValueError(f"expected CSI (B,3,114,10) or (B,1,3,114,10), got {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"expected three links/channels, got {tuple(x.shape)}")
    return x


def make_link_mask(batch: int, n_time: int, n_links: int, masked_links: int, device: torch.device) -> Tensor:
    if not 0 < masked_links < n_links:
        raise ValueError("masked_links must be between 1 and n_links - 1")
    link_ids = torch.stack([torch.randperm(n_links, device=device)[:masked_links] for _ in range(batch)])
    mask_links = torch.zeros(batch, n_links, dtype=torch.bool, device=device)
    mask_links.scatter_(1, link_ids, True)
    return mask_links[:, None, :].expand(batch, n_time, n_links).reshape(batch, n_time * n_links)


class EnvironmentAugment(nn.Module):
    """Style-changing CSI perturbations that preserve the pose signal."""

    def __init__(self, gain_std: float = 0.12, offset_std: float = 0.04,
                 subcarrier_drop: float = 0.08, time_jitter: int = 1) -> None:
        super().__init__()
        self.gain_std = float(gain_std)
        self.offset_std = float(offset_std)
        self.subcarrier_drop = float(subcarrier_drop)
        self.time_jitter = int(time_jitter)

    def forward(self, x: Tensor) -> Tensor:
        x = _squeeze_csi(x)
        batch, links, subcarriers, time = x.shape
        gain = torch.exp(torch.randn(batch, links, 1, 1, device=x.device, dtype=x.dtype) * self.gain_std)
        offset = torch.randn(batch, links, 1, 1, device=x.device, dtype=x.dtype) * self.offset_std
        out = x * gain + offset
        if self.subcarrier_drop > 0:
            keep = torch.rand(batch, 1, subcarriers, 1, device=x.device) > self.subcarrier_drop
            out = out * keep.to(dtype=out.dtype)
        if self.time_jitter > 0 and time > 1:
            shifts = torch.randint(-self.time_jitter, self.time_jitter + 1, (batch,), device=x.device)
            out = torch.stack([sample.roll(int(shift.item()), dims=-1) for sample, shift in zip(out, shifts)], dim=0)
        return out


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: Tensor, scale: float) -> Tensor:
    return GradientReversalFn.apply(x, scale)


class StructuredEncoder(nn.Module):
    """Time-link token encoder with explicit content/style factorization."""

    def __init__(self, *, n_subcarriers: int = 114, n_time: int = 10, n_links: int = 3,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6,
                 ffn_dim: int = 1536) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.n_time, self.n_links = int(n_time), int(n_links)
        self.n_tokens, self.embed_dim = self.n_time * self.n_links, int(embed_dim)
        self.token_proj = nn.Linear(n_subcarriers, embed_dim)
        self.time_pos = nn.Parameter(torch.zeros(1, n_time, 1, embed_dim))
        self.link_pos = nn.Parameter(torch.zeros(1, 1, n_links, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.content = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.style = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.link_pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def tokenize(self, x: Tensor) -> Tensor:
        x = _squeeze_csi(x)
        tokens = x.permute(0, 3, 1, 2).reshape(x.shape[0], self.n_tokens, -1)
        tokens = self.token_proj(tokens)
        pos = (self.time_pos + self.link_pos).reshape(1, self.n_tokens, self.embed_dim)
        return tokens + pos

    def encode_tokens(self, tokens: Tensor, visible_ids: Optional[Tensor] = None) -> Tensor:
        if visible_ids is not None:
            gather = visible_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            tokens = torch.gather(tokens, 1, gather)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return self.norm(self.blocks(torch.cat([cls, tokens], dim=1)))

    def forward(self, x: Tensor, *, visible_ids: Optional[Tensor] = None, return_tokens: bool = False):
        encoded = self.encode_tokens(self.tokenize(x), visible_ids=visible_ids)
        cls, token_states = encoded[:, 0], encoded[:, 1:]
        content = self.content(cls)
        style = self.style(token_states.mean(dim=1))
        if return_tokens:
            return {"encoded": encoded, "cls": cls, "content": content, "style": style, "tokens": token_states}
        return content, style


class MaskedTokenPredictor(nn.Module):
    def __init__(self, embed_dim: int, n_tokens: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, context: Tensor, visible: Tensor, visible_ids: Tensor) -> Tensor:
        batch, n_tokens, dim = context.shape[0], self.position.shape[1] - 1, visible.shape[-1]
        full = self.mask_token.expand(batch, n_tokens, dim).clone()
        full.scatter_(1, visible_ids.unsqueeze(-1).expand(-1, -1, dim), visible)
        sequence = torch.cat([context[:, None], full], dim=1) + self.position
        return self.norm(self.blocks(sequence))[:, 1:]


class EnvironmentDiscriminator(nn.Module):
    def __init__(self, embed_dim: int, n_environments: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Linear(embed_dim // 2, n_environments))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class PretrainOutput:
    loss: Tensor
    jepa_loss: Tensor
    invariance_loss: Tensor
    environment_loss: Tensor
    temporal_loss: Tensor
    content: Tensor
    style: Tensor


class EILJEPA(nn.Module):
    """Environment-Invariant Link-JEPA for cross-environment MM-Fi pose."""

    def __init__(self, *, n_subcarriers: int = 114, n_time: int = 10, n_links: int = 3,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6,
                 ffn_dim: int = 1536, predictor_depth: int = 2, masked_links: int = 1,
                 n_environments: int = 4, invariant_weight: float = 1.0,
                 environment_weight: float = 0.05, temporal_weight: float = 0.1,
                 grl_scale: float = 1.0) -> None:
        super().__init__()
        self.context_encoder = StructuredEncoder(n_subcarriers=n_subcarriers, n_time=n_time,
            n_links=n_links, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, ffn_dim=ffn_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = MaskedTokenPredictor(embed_dim, self.context_encoder.n_tokens,
            predictor_depth, num_heads)
        self.environment_augment = EnvironmentAugment()
        self.environment_head = EnvironmentDiscriminator(embed_dim, n_environments)
        self.masked_links = int(masked_links)
        self.invariant_weight = float(invariant_weight)
        self.environment_weight = float(environment_weight)
        self.temporal_weight = float(temporal_weight)
        self.grl_scale = float(grl_scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)

    def _masked_jepa(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        mask = make_link_mask(batch, self.context_encoder.n_time, self.context_encoder.n_links,
                              self.masked_links, x.device)
        visible_ids = (~mask).nonzero(as_tuple=False).reshape(batch, -1, 2)[:, :, 1]
        context = self.context_encoder.encode_tokens(self.context_encoder.tokenize(x), visible_ids=visible_ids)
        predicted = self.predictor(context[:, 0], context[:, 1:], visible_ids)
        with torch.no_grad():
            target = self.target_encoder(x, return_tokens=True)["tokens"]
            target = F.layer_norm(target, (target.shape[-1],))
        return F.smooth_l1_loss(predicted[mask], target[mask])

    def forward(self, x: Tensor, *, environment_ids: Optional[Tensor] = None,
                next_x: Optional[Tensor] = None) -> PretrainOutput:
        x = _squeeze_csi(x)
        view_a, view_b = self.environment_augment(x), self.environment_augment(x)
        jepa_loss = self._masked_jepa(view_a)
        content_a, style_a = self.context_encoder(view_a)
        content_b, _ = self.context_encoder(view_b)
        invariance_loss = F.mse_loss(F.normalize(content_a, dim=-1), F.normalize(content_b, dim=-1))
        if environment_ids is None:
            environment_loss = x.new_zeros(())
        else:
            logits = self.environment_head(gradient_reverse(content_a, self.grl_scale))
            environment_loss = F.cross_entropy(logits, environment_ids)
        if next_x is None:
            temporal_loss = x.new_zeros(())
        else:
            next_content, _ = self.context_encoder(_squeeze_csi(next_x))
            temporal_loss = 1.0 - F.cosine_similarity(content_a, next_content, dim=-1).mean()
        loss = (jepa_loss + self.invariant_weight * invariance_loss
                 + self.environment_weight * environment_loss
                 + self.temporal_weight * temporal_loss)
        return PretrainOutput(loss=loss, jepa_loss=jepa_loss, invariance_loss=invariance_loss,
            environment_loss=environment_loss, temporal_loss=temporal_loss,
            content=content_a, style=style_a)

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}


class PoseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 51))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).reshape(x.shape[0], 17, 3)


class FactorizedPoseModel(nn.Module):
    """Absolute root from style-aware features; relative joints from invariant content."""

    encoder_readout = "factorized_content_style"

    def __init__(self, dropout: float = 0.1, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = StructuredEncoder(**encoder_kwargs)
        dim = self.encoder.embed_dim
        self.root_head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, 3))
        self.relative_head = PoseMLP(dim, hidden_dim=min(256, dim), dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        content, style = self.encoder(_squeeze_csi(x))
        root = self.root_head(torch.cat([content, style], dim=-1)).unsqueeze(1)
        relative = self.relative_head(content)
        relative = relative - relative[:, :1]
        return root + relative


# Alias retained while the paper-facing name is finalized.
EDCJEPA = EILJEPA
