from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return -ctx.strength * grad, None


def grad_reverse(x: Tensor, strength: float) -> Tensor:
    return _GradientReverse.apply(x, float(strength))


class SceneDecoupledEncoder(nn.Module):
    """Frequency-local, link-graph encoder with body/layout readouts."""

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        n_time: int = 10,
        n_links: int = 3,
        n_subcarriers: int = 114,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.n_time = int(n_time)
        self.n_links = int(n_links)
        self.n_subcarriers = int(n_subcarriers)
        self.n_tokens = self.n_time * self.n_links

        hidden = max(32, embed_dim // 2)
        self.spectral_stem = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden, embed_dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.raw_projection = nn.Linear(n_subcarriers, embed_dim)
        self.time_embed = nn.Parameter(torch.zeros(1, n_time, 1, embed_dim))
        self.link_embed = nn.Parameter(torch.zeros(1, 1, n_links, embed_dim))
        self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.link_embed, std=0.02)
        nn.init.trunc_normal_(self.global_token, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(
            temporal_layer, num_layers=max(1, depth // 2), enable_nested_tensor=False
        )
        graph_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_blocks = nn.TransformerEncoder(
            graph_layer, num_layers=max(1, depth - depth // 2), enable_nested_tensor=False
        )
        self.body_projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.layout_projection = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.body_token_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

    def _tokens(self, x: Tensor) -> Tensor:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 4:
            raise ValueError(f"expected CSI tensor (B,links,subcarriers,time), got {tuple(x.shape)}")
        b, links, subcarriers, times = x.shape
        if (links, subcarriers, times) != (self.n_links, self.n_subcarriers, self.n_time):
            raise ValueError(
                f"expected {(self.n_links, self.n_subcarriers, self.n_time)}, "
                f"got {(links, subcarriers, times)}"
            )
        # Robust per-frame normalization keeps scene amplitude scale available only
        # through the raw statistics that are concatenated into layout below.
        center = x.median(dim=2, keepdim=True).values
        scale = (x - center).abs().mean(dim=2, keepdim=True).clamp_min(1e-3)
        normalized = (x - center) / scale
        spectral = normalized.permute(0, 3, 1, 2).reshape(b * times * links, 1, subcarriers)
        local = self.spectral_stem(spectral).flatten(1)
        raw = self.raw_projection(normalized.permute(0, 3, 1, 2).reshape(b * times * links, subcarriers))
        tokens = (local + raw).view(b, times, links, self.embed_dim)
        tokens = tokens + self.time_embed + self.link_embed
        return tokens

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        tokens = self._tokens(x)
        b = tokens.shape[0]
        temporal = tokens.permute(0, 2, 1, 3).reshape(b * self.n_links, self.n_time, self.embed_dim)
        temporal = self.temporal_blocks(temporal).view(b, self.n_links, self.n_time, self.embed_dim)
        link_nodes = temporal.mean(dim=2)
        global_node = link_nodes.mean(dim=1, keepdim=True) + self.global_token
        graph_input = torch.cat([global_node, link_nodes], dim=1)
        graph = self.output_norm(self.graph_blocks(graph_input))
        graph_global = graph[:, 0]
        graph_links = graph[:, 1:]
        body_tokens = temporal + graph_links[:, :, None, :]
        body_tokens = self.body_token_norm(body_tokens).reshape(b, self.n_tokens, self.embed_dim)
        body_global = self.body_projection(graph_global)
        layout_global = self.layout_projection(graph_global)
        return body_tokens, body_global, layout_global


class SceneDecoupledJEPA(nn.Module):
    model_variant = "scene_decoupled_geometric_jepa_v1"

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        n_time: int = 10,
        n_links: int = 3,
        n_subcarriers: int = 114,
        scene_count: int = 3,
        predictor_depth: int = 1,
        lambda_body: float = 1.0,
        lambda_adv: float = 0.05,
        lambda_layout: float = 0.10,
        lambda_orth: float = 0.01,
        mask_links: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_kwargs = dict(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
            n_subcarriers=n_subcarriers, dropout=dropout,
        )
        self.encoder = SceneDecoupledEncoder(**encoder_kwargs)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            *[layer for _ in range(max(0, predictor_depth - 1)) for layer in (nn.Linear(embed_dim, embed_dim), nn.GELU())],
            nn.Linear(embed_dim, embed_dim),
        )
        self.scene_body_head = nn.Linear(embed_dim, scene_count)
        self.scene_layout_head = nn.Linear(embed_dim, scene_count)
        self.lambda_body = float(lambda_body)
        self.lambda_adv = float(lambda_adv)
        self.lambda_layout = float(lambda_layout)
        self.lambda_orth = float(lambda_orth)
        self.mask_links = int(mask_links)
        self.n_links = int(n_links)
        self.n_time = int(n_time)
        self.n_tokens = self.n_links * self.n_time
        if not 0 < self.mask_links < self.n_links:
            raise ValueError("mask_links must be between 1 and n_links - 1")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)

    def _view(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b = x.shape[0]
        link_order = torch.rand(b, self.n_links, device=x.device).argsort(dim=1)
        mask = torch.zeros(b, self.n_links, dtype=torch.bool, device=x.device)
        mask.scatter_(1, link_order[:, : self.mask_links], True)
        viewed = x.clone()
        viewed[mask[:, :, None, None].expand_as(viewed)] = 0.0
        scale = 0.9 + 0.2 * torch.rand(b, 1, 1, 1, device=x.device)
        noise = torch.randn_like(viewed) * viewed.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3) * 0.02
        return viewed * scale + noise, mask

    @staticmethod
    def _cross_covariance(a: Tensor, b: Tensor) -> Tensor:
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        return (a.transpose(0, 1) @ b) / max(1, a.shape[0] - 1)

    def forward(self, x: Tensor, scene: Tensor) -> tuple[Tensor, dict]:
        view1, mask1 = self._view(x)
        view2, _ = self._view(x)
        body1, global1, layout1 = self.encoder(view1)
        body2, global2, _ = self.encoder(view2)
        with torch.no_grad():
            target_tokens, target_body, _ = self.target_encoder(x)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
            target_body = F.layer_norm(target_body, (target_body.shape[-1],))
        predicted = self.predictor(body1)
        token_mask = mask1[:, :, None].expand(x.shape[0], self.n_links, self.n_time).reshape(x.shape[0], self.n_tokens)
        masked_loss = F.smooth_l1_loss(predicted[token_mask], target_tokens[token_mask])
        body_consistency = (1.0 - F.cosine_similarity(global1, global2, dim=-1)).mean()
        body_logits = self.scene_body_head(grad_reverse(global1, self.lambda_adv))
        layout_logits = self.scene_layout_head(layout1)
        body_scene = F.cross_entropy(body_logits, scene)
        layout_scene = F.cross_entropy(layout_logits, scene)
        cross_cov = self._cross_covariance(global1, layout1).pow(2).mean()
        loss = masked_loss + self.lambda_body * body_consistency + self.lambda_adv * body_scene + self.lambda_layout * layout_scene + self.lambda_orth * cross_cov
        return loss, {
            "loss": float(loss.detach()),
            "masked_loss": float(masked_loss.detach()),
            "body_consistency": float(body_consistency.detach()),
            "body_scene_acc": float((body_logits.argmax(dim=1) == scene).float().mean().detach()),
            "layout_scene_acc": float((layout_logits.argmax(dim=1) == scene).float().mean().detach()),
            "orth_loss": float(cross_cov.detach()),
            "masked_tokens": int(token_mask.sum()),
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.encoder.state_dict().items()}


class SceneDecoupledPoseModel(nn.Module):
    """Joint-query decoder using body tokens and a gated layout root branch."""

    encoder_readout = "body_layout_joint_query"

    def __init__(
        self,
        n_joints: int = 17,
        query_depth: int = 2,
        query_heads: int = 6,
        dropout_p: float = 0.0,
        layout_gate_init: float = -2.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = SceneDecoupledEncoder(**encoder_kwargs)
        dim = self.encoder.embed_dim
        if dim % query_heads:
            raise ValueError("query_heads must divide encoder embedding dimension")
        self.query_embed = nn.Parameter(torch.zeros(1, n_joints, dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=query_heads, dim_feedforward=dim * 4,
            dropout=dropout_p, activation="gelu", batch_first=True, norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=query_depth)
        self.query_norm = nn.LayerNorm(dim)
        self.relative_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.root_head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 3))
        self.layout_residual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.layout_gate = nn.Parameter(torch.tensor(float(layout_gate_init)))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        body_tokens, body_global, layout_global = self.encoder(x)
        batch = x.shape[0]
        queries = self.query_embed.expand(batch, -1, -1)
        decoded = self.query_norm(self.query_decoder(queries, body_tokens))
        relative = self.relative_head(decoded)
        relative = relative - relative[:, :1]
        root_context = torch.cat([body_global, layout_global], dim=-1)
        root = self.root_head(root_context) + torch.sigmoid(self.layout_gate) * self.layout_residual(layout_global)
        return relative + root[:, None, :]


def build_encoder_kwargs(model_cfg: dict, input_cfg: dict) -> dict:
    return {
        "embed_dim": int(model_cfg["embed_dim"]),
        "depth": int(model_cfg["depth"]),
        "num_heads": int(model_cfg["num_heads"]),
        "ffn_dim": int(model_cfg["ffn_dim"]),
        "n_time": int(input_cfg["time"]),
        "n_links": int(input_cfg["links"]),
        "n_subcarriers": int(input_cfg["subcarriers"]),
        "dropout": float(model_cfg.get("dropout", 0.0)),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SceneDecoupledJEPA(embed_dim=48, depth=2, num_heads=6, ffn_dim=96)
    x = torch.randn(4, 3, 114, 10)
    loss, log = model(x, torch.tensor([0, 1, 2, 0]))
    assert torch.isfinite(loss)
    pose = SceneDecoupledPoseModel(embed_dim=48, depth=2, num_heads=6, ffn_dim=96)
    output = pose(x)
    assert output.shape == (4, 17, 3)
    print({"loss": float(loss), "pose_shape": tuple(output.shape), **log})
