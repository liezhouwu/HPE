from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from WIFIJEPA.src.model import Predictor, StructuredViTEncoder
from pose_ssl.model import PoseHead


def make_temporal_mask(
    batch_size: int,
    window_frames: int,
    n_links: int,
    n_time: int,
    masked_links: int,
    future_frames: int,
    mode: str,
    device: torch.device,
) -> Tensor:
    if not 0 < masked_links < n_links:
        raise ValueError("masked_links must be between 1 and n_links - 1")
    if not 0 < future_frames < window_frames:
        raise ValueError("future_frames must be between 1 and window_frames - 1")
    if mode not in {"link", "future", "joint"}:
        raise ValueError("mask mode must be link, future, or joint")

    links = torch.rand(batch_size, n_links, device=device).argsort(dim=1)[:, :masked_links]
    link_mask = torch.zeros(batch_size, n_links, dtype=torch.bool, device=device)
    link_mask.scatter_(1, links, True)
    link_mask = link_mask[:, None, :].expand(batch_size, window_frames, n_links)

    future_mask = torch.zeros(batch_size, window_frames, n_links, dtype=torch.bool, device=device)
    future_mask[:, -future_frames:] = True
    if mode == "link":
        mask = link_mask
    elif mode == "future":
        mask = future_mask
    else:
        mask = link_mask | future_mask
    return mask[:, :, None, :].expand(batch_size, window_frames, n_time, n_links).reshape(
        batch_size, window_frames * n_time * n_links
    )


def _masked_prediction_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        raise ValueError("JEPA mask selected no target tokens")
    return F.smooth_l1_loss(prediction[mask], target[mask])


class LinkFactorizedJEPA(nn.Module):
    """JEPA with shared-latent prediction and a separately trained residual head."""

    model_variant = "factorized_link_shared_v1"

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 6,
        ffn_dim: int = 3072,
        predictor_depth: int = 2,
        n_time: int = 10,
        n_links: int = 3,
        n_masked_links: int = 2,
        residual_loss_weight: float = 1.0,
        shared_std_floor: float = 0.25,
    ):
        super().__init__()
        self.context_encoder = StructuredViTEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.shared_projection = nn.Linear(embed_dim, embed_dim)
        self.target_shared_projection = copy.deepcopy(self.shared_projection)
        self.shared_predictor = Predictor(
            embed_dim=embed_dim, depth=predictor_depth, num_heads=num_heads,
            n_tokens=n_time * n_links,
        )
        self.residual_predictor = Predictor(
            embed_dim=embed_dim, depth=predictor_depth, num_heads=num_heads,
            n_tokens=n_time * n_links,
        )
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_shared_projection.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.n_time = n_time
        self.n_links = n_links
        self.n_masked_links = n_masked_links
        self.residual_loss_weight = float(residual_loss_weight)
        self.shared_std_floor = float(shared_std_floor)
        if self.residual_loss_weight < 0 or self.shared_std_floor < 0:
            raise ValueError("loss weights and shared_std_floor must be non-negative")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_shared_projection.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)
        for online, target in zip(self.shared_projection.parameters(), self.target_shared_projection.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)

    def forward(self, x: Tensor, target_mask: Tensor | None = None):
        batch = x.shape[0]
        if target_mask is None:
            from WIFIJEPA.src.masking import make_link_mask

            target_mask = make_link_mask(
                batch, self.n_time, self.n_links, self.n_masked_links, x.device
            )
        visible_mask = ~target_mask
        cls, visible, visible_ids = self.context_encoder.forward_visible(x, visible_mask)
        shared_cls = self.shared_projection(cls)
        shared_visible = self.shared_projection(visible)
        shared_prediction = self.shared_predictor(shared_cls, shared_visible, visible_ids)

        residual_prediction = self.residual_predictor(
            cls.detach(), visible.detach(), visible_ids
        )
        with torch.no_grad():
            target = self.target_encoder(x, return_tokens=True)[:, 1:]
            target = F.layer_norm(target, (target.shape[-1],))
            shared_target = self.target_shared_projection(target)
            shared_target = shared_target.view(batch, self.n_time, self.n_links, -1)
            shared_consensus = shared_target.mean(dim=2, keepdim=True).expand_as(shared_target)
            shared_consensus = shared_consensus.reshape(batch, -1, shared_consensus.shape[-1])
            residual_target = target - shared_consensus

        shared_loss = _masked_prediction_loss(shared_prediction, shared_consensus, target_mask)
        per_sample = (shared_prediction * target_mask.unsqueeze(-1)).sum(dim=1)
        per_sample = per_sample / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
        if batch > 1 and self.shared_std_floor:
            shared_variance_loss = F.relu(
                self.shared_std_floor - per_sample.std(dim=0, unbiased=False)
            ).mean()
        else:
            shared_variance_loss = shared_loss.new_zeros(())
        residual_loss = _masked_prediction_loss(residual_prediction, residual_target, target_mask)
        loss = shared_loss + shared_variance_loss + self.residual_loss_weight * residual_loss
        return loss, {
            "loss": float(loss.detach()),
            "shared_loss": float(shared_loss.detach()),
            "shared_variance_loss": float(shared_variance_loss.detach()),
            "residual_loss": float(residual_loss.detach()),
            "masked_tokens": int(target_mask.sum()),
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}

    def export_shared_projection_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.shared_projection.state_dict().items()}


class TemporalLinkJEPA(nn.Module):
    """Window JEPA: mask a future interval, selected links, or their union."""

    model_variant = "temporal_link_jepa_v1"

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 6,
        ffn_dim: int = 3072,
        predictor_depth: int = 2,
        n_time: int = 10,
        n_links: int = 3,
        window_frames: int = 6,
        future_frames: int = 2,
        masked_links: int = 1,
        mask_mode: str = "joint",
    ):
        super().__init__()
        if window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        self.context_encoder = StructuredViTEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.window_frames = int(window_frames)
        self.future_frames = int(future_frames)
        self.n_time = int(n_time)
        self.n_links = int(n_links)
        self.masked_links = int(masked_links)
        self.mask_mode = str(mask_mode)
        self.tokens_per_frame = self.n_time * self.n_links
        self.predictor = Predictor(
            embed_dim=embed_dim, depth=predictor_depth, num_heads=num_heads,
            n_tokens=self.window_frames * self.tokens_per_frame,
        )
        if not 0 < self.future_frames < self.window_frames:
            raise ValueError("future_frames must be between 1 and window_frames - 1")
        if self.mask_mode not in {"link", "future", "joint"}:
            raise ValueError("mask_mode must be link, future, or joint")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)

    def forward(self, windows: Tensor, target_mask: Tensor | None = None):
        if windows.ndim != 5 or windows.shape[1] != self.window_frames:
            raise ValueError(
                f"expected windows (B,{self.window_frames},links,subcarriers,time), "
                f"got {tuple(windows.shape)}"
            )
        batch = windows.shape[0]
        if target_mask is None:
            target_mask = make_temporal_mask(
                batch, self.window_frames, self.n_links, self.n_time,
                self.masked_links, self.future_frames, self.mask_mode, windows.device,
            )
        target_mask = target_mask.view(batch, self.window_frames, self.tokens_per_frame)
        context_tokens = []
        visible_ids = []
        cls_tokens = []
        cls_available = []
        for frame_index in range(self.window_frames):
            frame_mask = target_mask[:, frame_index].view(batch, self.n_time, self.n_links)
            cls, visible, ids = self.context_encoder.forward_visible(
                windows[:, frame_index], ~frame_mask.reshape(batch, -1)
            )
            context_tokens.append(visible)
            visible_ids.append(ids + frame_index * self.tokens_per_frame)
            cls_tokens.append(cls)
            cls_available.append((~frame_mask.reshape(batch, -1)).any(dim=1))
        visible = torch.cat(context_tokens, dim=1)
        ids = torch.cat(visible_ids, dim=1)
        cls_stack = torch.stack(cls_tokens, dim=1)
        cls_weights = torch.stack(cls_available, dim=1).unsqueeze(-1)
        cls = (cls_stack * cls_weights).sum(dim=1) / cls_weights.sum(dim=1).clamp_min(1)
        prediction = self.predictor(cls, visible, ids)
        with torch.no_grad():
            target = self.target_encoder(
                windows.reshape(batch * self.window_frames, *windows.shape[2:]),
                return_tokens=True,
            )[:, 1:]
            target = F.layer_norm(target, (target.shape[-1],))
            target = target.view(batch, self.window_frames * self.tokens_per_frame, -1)
        flat_mask = target_mask.reshape(batch, -1)
        loss = _masked_prediction_loss(prediction, target, flat_mask)
        return loss, {
            "loss": float(loss.detach()),
            "masked_tokens": int(flat_mask.sum()),
            "future_frames": self.future_frames,
            "mask_mode": self.mask_mode,
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}


class TemporalWindowDataset(torch.utils.data.Dataset):
    """Sequence-safe CSI windows built from adjacent pose-frame CSI files."""

    def __init__(
        self,
        dataset_root: str,
        sequence_keys,
        *,
        context_frames: int,
        future_frames: int,
        protocol: str,
        split: str,
        scope: str,
        manifest_fingerprint: str,
    ):
        from pose_ssl.metafi.pretrain_data import MetaFiPretrainDataset

        if context_frames < 1 or future_frames < 1:
            raise ValueError("context_frames and future_frames must be positive")
        self.context_frames = int(context_frames)
        self.future_frames = int(future_frames)
        self.offsets = tuple(range(-self.context_frames + 1, self.future_frames + 1))
        neighbor_offsets = tuple(offset for offset in self.offsets if offset != 0)
        self.base = MetaFiPretrainDataset(
            dataset_root, sequence_keys, neighbor_offsets=neighbor_offsets,
            protocol=protocol, split=split, scope=scope,
            manifest_fingerprint=manifest_fingerprint,
        )
        lower, upper = min(self.offsets), max(self.offsets)
        self.records = tuple(
            index for index, record in enumerate(self.base._records)
            if 0 <= record.frame_index + lower < 297
            and 0 <= record.frame_index + upper < 297
        )
        if not self.records:
            raise ValueError("no complete temporal windows remain inside selected sequences")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tensor:
        sample = self.base[self.records[index]]
        frames = {sample.frame_index: sample.anchor}
        frames.update(zip(sample.neighbor_indices, sample.neighbors, strict=True))
        indices = tuple(sample.frame_index + offset for offset in self.offsets)
        if any(frame_index not in frames for frame_index in indices):
            raise RuntimeError("temporal window is missing a same-sequence CSI frame")
        return torch.stack([frames[frame_index] for frame_index in indices])


class SharedStructuredPoseModel(nn.Module):
    """Downstream model whose readout uses only the link-shared projection."""

    def __init__(self, dropout_p: float = 0.3, **encoder_kwargs):
        super().__init__()
        encoder_kwargs.pop("predictor_depth", None)
        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        self.shared_projection = nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        self.encoder_readout = "shared_mean"
        self.head = PoseHead(self.encoder.out_dim, dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        token_states = self.encoder(x, return_tokens=True)[:, 1:]
        shared = self.shared_projection(token_states)
        batch = x.shape[0]
        shared = shared.view(batch, self.encoder.tokenizer.n_time,
                             self.encoder.tokenizer.n_links, -1)
        return self.head(shared.mean(dim=(1, 2)))
