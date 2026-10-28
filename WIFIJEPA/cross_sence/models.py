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

    model_variant = "factorized_link_shared_v4"

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
        residual_loss_weight: float = 0.1,
        shared_std_floor: float = 0.25,
        shared_latent_norm: bool = True,
        separate_residual_projection: bool = False,
        residual_std_floor: float = 0.1,
    ):
        super().__init__()
        self.context_encoder = StructuredViTEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            ffn_dim=ffn_dim, n_time=n_time, n_links=n_links,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.shared_projection = nn.Linear(embed_dim, embed_dim)
        # Start from the encoder latent space; the projection can still learn
        # a task-specific shared subspace during pretraining.
        nn.init.eye_(self.shared_projection.weight)
        nn.init.zeros_(self.shared_projection.bias)
        self.target_shared_projection = copy.deepcopy(self.shared_projection)
        self.separate_residual_projection = bool(separate_residual_projection)
        if self.separate_residual_projection:
            self.residual_projection = nn.Linear(embed_dim, embed_dim)
            nn.init.eye_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)
            self.target_residual_projection = copy.deepcopy(self.residual_projection)
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
        if self.separate_residual_projection:
            for parameter in self.target_residual_projection.parameters():
                parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.n_time = n_time
        self.n_links = n_links
        self.n_masked_links = n_masked_links
        self.residual_loss_weight = float(residual_loss_weight)
        self.shared_std_floor = float(shared_std_floor)
        self.shared_latent_norm = bool(shared_latent_norm)
        self.residual_std_floor = float(residual_std_floor)
        if self.residual_loss_weight < 0 or self.shared_std_floor < 0 or self.residual_std_floor < 0:
            raise ValueError("loss weights and shared_std_floor must be non-negative")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        self.target_shared_projection.eval()
        if self.separate_residual_projection:
            self.target_residual_projection.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)
        for online, target in zip(self.shared_projection.parameters(), self.target_shared_projection.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)
        if self.separate_residual_projection:
            for online, target in zip(self.residual_projection.parameters(), self.target_residual_projection.parameters(), strict=True):
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
        if self.shared_latent_norm:
            shared_cls = F.layer_norm(shared_cls, (shared_cls.shape[-1],))
            shared_visible = F.layer_norm(shared_visible, (shared_visible.shape[-1],))
        shared_prediction = self.shared_predictor(shared_cls, shared_visible, visible_ids)

        # Predict the link-specific component in the same projected latent
        # space as the shared branch.  This keeps the factorization coherent
        # and lets the residual objective shape the exported encoder.
        if self.separate_residual_projection:
            residual_cls = self.residual_projection(cls)
            residual_visible = self.residual_projection(visible)
            residual_cls = F.layer_norm(residual_cls, (residual_cls.shape[-1],))
            residual_visible = F.layer_norm(residual_visible, (residual_visible.shape[-1],))
            residual_prediction = self.residual_predictor(
                residual_cls, residual_visible, visible_ids
            )
        else:
            residual_prediction = self.residual_predictor(
                shared_cls, shared_visible, visible_ids
            )
        with torch.no_grad():
            target = self.target_encoder(x, return_tokens=True)[:, 1:]
            target = F.layer_norm(target, (target.shape[-1],))
            shared_target = self.target_shared_projection(target)
            if self.shared_latent_norm:
                shared_target = F.layer_norm(shared_target, (shared_target.shape[-1],))
            shared_target = shared_target.view(batch, self.n_time, self.n_links, -1)
            shared_consensus = shared_target.mean(dim=2, keepdim=True)
            residual_target = (shared_target - shared_consensus).reshape(
                batch, -1, shared_target.shape[-1]
            )
            if self.separate_residual_projection:
                residual_target = self.target_residual_projection(target)
                residual_target = F.layer_norm(residual_target, (residual_target.shape[-1],))
                residual_target = residual_target.view(batch, self.n_time, self.n_links, -1)
                residual_target = (
                    residual_target - residual_target.mean(dim=2, keepdim=True)
                ).reshape(batch, -1, residual_target.shape[-1])
            shared_target = shared_target.reshape(batch, -1, shared_target.shape[-1])

        # Preserve link-specific information for pose recovery.  The shared
        # branch is now an auxiliary regularizer instead of replacing every
        # link target with its mean.
        link_loss = _masked_prediction_loss(shared_prediction, shared_target, target_mask)
        residual_loss = _masked_prediction_loss(residual_prediction, residual_target, target_mask)
        predicted_tokens = shared_prediction.view(batch, self.n_time, self.n_links, -1)
        predicted_consensus = predicted_tokens.mean(dim=2, keepdim=True).expand_as(
            predicted_tokens
        ).reshape(batch, -1, predicted_tokens.shape[-1])
        target_consensus = shared_target.view(batch, self.n_time, self.n_links, -1).mean(
            dim=2, keepdim=True
        ).expand_as(predicted_tokens).reshape(batch, -1, predicted_tokens.shape[-1])
        consensus_loss = _masked_prediction_loss(predicted_consensus, target_consensus, target_mask)
        per_sample = (shared_prediction * target_mask.unsqueeze(-1)).sum(dim=1)
        per_sample = per_sample / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
        if batch > 1 and self.shared_std_floor:
            shared_variance_loss = F.relu(
                self.shared_std_floor - per_sample.std(dim=0, unbiased=False)
            ).mean()
        else:
            shared_variance_loss = link_loss.new_zeros(())
        if batch > 1 and self.separate_residual_projection and self.residual_std_floor:
            residual_per_sample = (residual_prediction * target_mask.unsqueeze(-1)).sum(dim=1)
            residual_per_sample = residual_per_sample / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
            residual_variance_loss = F.relu(
                self.residual_std_floor - residual_per_sample.std(dim=0, unbiased=False)
            ).mean()
        else:
            residual_variance_loss = link_loss.new_zeros(())
        loss = (
            link_loss + 0.1 * consensus_loss + shared_variance_loss
            + residual_variance_loss + self.residual_loss_weight * residual_loss
        )
        return loss, {
            "loss": float(loss.detach()),
            "shared_loss": float(link_loss.detach()),
            "consensus_loss": float(consensus_loss.detach()),
            "shared_variance_loss": float(shared_variance_loss.detach()),
            "residual_variance_loss": float(residual_variance_loss.detach()),
            "residual_loss": float(residual_loss.detach()),
            "masked_tokens": int(target_mask.sum()),
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}

    def export_shared_projection_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.shared_projection.state_dict().items()}


class TemporalLinkJEPA(nn.Module):
    """Stable v2 temporal JEPA with a fixed joint masking distribution."""

    model_variant = "temporal_link_jepa_v7"

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
        temporal_depth: int = 1,
        temporal_readout_input: str = "masked",
        link_mask_probability: float = 0.0,
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
        self.temporal_readout_input = str(temporal_readout_input)
        if self.temporal_readout_input != "masked":
            raise ValueError(
                "temporal_readout_input must be 'masked' for temporal_link_jepa_v7"
            )
        self.link_mask_probability = float(link_mask_probability)
        if not 0.0 <= self.link_mask_probability <= 1.0:
            raise ValueError("link_mask_probability must be between 0 and 1")
        if self.link_mask_probability != 0.0:
            raise ValueError(
                "temporal_link_jepa_v7 uses the fixed v2 joint mask; "
                "link_mask_probability must be 0"
            )
        self.tokens_per_frame = self.n_time * self.n_links
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.masked_frame_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, self.window_frames + 1, embed_dim)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(
            temporal_layer, num_layers=int(temporal_depth), enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.masked_frame_cls, std=0.02)
        nn.init.trunc_normal_(self.temporal_query, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
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
        sampled_mode = self.mask_mode
        if target_mask is None:
            target_mask = make_temporal_mask(
                batch, self.window_frames, self.n_links, self.n_time,
                self.masked_links, self.future_frames, sampled_mode, windows.device,
            )
        target_mask = target_mask.view(batch, self.window_frames, self.tokens_per_frame)
        visible_mask = ~target_mask
        frames_have_visible = visible_mask.any(dim=2)

        # Encode all usable frames as one larger batch.  The old implementation
        # launched one transformer call per frame and also encoded fully masked
        # future frames whose CLS values were discarded.  In joint mode those
        # future calls are replaced by a learned frame placeholder.
        common_frame_pattern = torch.equal(
            frames_have_visible,
            frames_have_visible[:1].expand_as(frames_have_visible),
        )
        if common_frame_pattern:
            active_frames = frames_have_visible[0].nonzero(as_tuple=False).flatten()
        else:
            active_frames = torch.arange(self.window_frames, device=windows.device)

        active_windows = windows[:, active_frames]
        active_masks = target_mask[:, active_frames]
        active_batch = active_windows.reshape(
            batch * active_frames.numel(), *windows.shape[2:]
        )
        tokens = self.context_encoder.tokenizer(active_batch)
        active_mask = active_masks.reshape(batch * active_frames.numel(), -1)
        mask_token = self.mask_token.to(dtype=tokens.dtype).expand_as(tokens)
        tokens = torch.where(active_mask.unsqueeze(-1), mask_token, tokens)
        encoded = self.context_encoder._encode(tokens)
        active_cls = encoded[:, 0].view(batch, active_frames.numel(), -1)
        active_tokens = encoded[:, 1:].view(
            batch, active_frames.numel(), self.tokens_per_frame, -1
        )

        cls_stack = self.masked_frame_cls.to(dtype=active_cls.dtype).expand(
            batch, self.window_frames, -1
        ).clone()
        all_tokens = active_tokens.new_zeros(
            batch, self.window_frames, self.tokens_per_frame, active_tokens.shape[-1]
        )
        if common_frame_pattern:
            cls_stack[:, active_frames] = active_cls
            all_tokens[:, active_frames] = active_tokens
        else:
            # Fallback for custom masks whose visible-frame pattern differs by
            # sample; this path retains the previous semantics safely.
            cls_stack = active_cls.view(batch, self.window_frames, -1)
            all_tokens = active_tokens.view(
                batch, self.window_frames, self.tokens_per_frame, -1
            )

        visible = all_tokens[visible_mask].view(batch, -1, all_tokens.shape[-1])
        visible_coords = visible_mask.nonzero(as_tuple=False).reshape(batch, -1, 3)
        ids = (
            visible_coords[:, :, 1] * self.tokens_per_frame
            + visible_coords[:, :, 2]
        )
        # v2-compatible masked branch.  This remains the only JEPA objective.
        masked_sequence = torch.cat(
            [self.temporal_query.to(dtype=cls_stack.dtype).expand(batch, -1, -1), cls_stack],
            dim=1,
        ) + self.temporal_pos.to(dtype=cls_stack.dtype)
        masked_cls = self.temporal_norm(self.temporal_blocks(masked_sequence))[:, 0]

        prediction = self.predictor(masked_cls, visible, ids)
        # Match the target latent scale.  Without this normalization the
        # predictor can keep drifting in scale while the EMA target is nearly
        # frozen, which causes the late-epoch temporal loss rebound.
        prediction = F.layer_norm(prediction, (prediction.shape[-1],))
        with torch.no_grad():
            target = self.target_encoder(
                windows.reshape(batch * self.window_frames, *windows.shape[2:]),
                return_tokens=True,
            )[:, 1:]
            target = F.layer_norm(target, (target.shape[-1],))
            target = target.view(batch, self.window_frames * self.tokens_per_frame, -1)
        flat_mask = target_mask.reshape(batch, -1)
        masked_loss = _masked_prediction_loss(prediction, target, flat_mask)
        return masked_loss, {
            "loss": float(masked_loss.detach()),
            "masked_loss": float(masked_loss.detach()),
            "masked_tokens": int(flat_mask.sum()),
            "future_frames": self.future_frames,
            "mask_mode": sampled_mode,
            "link_mask_batch": 0,
        }

    def export_encoder_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.context_encoder.state_dict().items()}

    def export_temporal_state_dict(self) -> dict[str, Tensor]:
        """Export the temporal readout used by the temporal downstream model."""
        state = self.state_dict()
        return {
            name: value.detach().clone()
            for name, value in state.items()
            if name in {"temporal_query", "temporal_pos"}
            or name.startswith("temporal_blocks.")
            or name.startswith("temporal_norm.")
        }


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


class TemporalWindowPoseDataset(torch.utils.data.Dataset):
    """Temporal windows with the anchor-frame pose target for downstream use."""

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
        from WIFIJEPA.src.data import WiFiJEPAData, _load_ground_truth

        if context_frames < 1 or future_frames < 1:
            raise ValueError("context_frames and future_frames must be positive")
        self.context_frames = int(context_frames)
        self.future_frames = int(future_frames)
        self.offsets = tuple(range(-self.context_frames + 1, self.future_frames + 1))
        neighbor_offsets = tuple(offset for offset in self.offsets if offset != 0)
        self.base = WiFiJEPAData(
            dataset_root, sequence_keys, neighbor_offsets=neighbor_offsets,
            protocol=protocol, split=split, scope=scope,
            manifest_fingerprint=manifest_fingerprint,
        )
        self._load_ground_truth = _load_ground_truth
        lower, upper = min(self.offsets), max(self.offsets)
        self.records = tuple(
            index for index, record in enumerate(self.base._records)
            if 0 <= record.frame_index + lower < 297
            and 0 <= record.frame_index + upper < 297
        )
        if not self.records:
            raise ValueError("no complete temporal pose windows remain")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.base._records[self.records[index]]
        sample = self.base[self.records[index]]
        frames = {sample.frame_index: sample.anchor}
        frames.update(zip(sample.neighbor_indices, sample.neighbors, strict=True))
        indices = tuple(sample.frame_index + offset for offset in self.offsets)
        if any(frame_index not in frames for frame_index in indices):
            raise RuntimeError("temporal pose window is missing a same-sequence CSI frame")
        gt_path = (
            self.base.dataset_root / record.sequence_key.scene / record.sequence_key.subject
            / record.sequence_key.action / "ground_truth.npy"
        )
        pose = torch.as_tensor(
            self._load_ground_truth(str(gt_path))[sample.frame_index].copy(),
            dtype=torch.float32,
        )
        return {
            "modality": ["wifi-csi"],
            "input_wifi-csi": torch.stack([frames[frame_index] for frame_index in indices]),
            "output": pose,
            "scene": record.sequence_key.scene,
            "subject": record.sequence_key.subject,
            "action": record.sequence_key.action,
            "idx": int(sample.frame_index),
        }


class TemporalPoseModel(nn.Module):
    """Pose regressor using the pretrained anchor-plus-temporal representation."""

    encoder_readout = "temporal"

    def __init__(self, dropout_p: float = 0.3, window_frames: int = 6,
                 temporal_depth: int = 1, anchor_frame: int | None = None,
                 temporal_gate_init: float = -2.2, **encoder_kwargs):
        super().__init__()
        encoder_kwargs.pop("predictor_depth", None)
        self.window_frames = int(window_frames)
        self.anchor_frame = (
            int(anchor_frame) if anchor_frame is not None else self.window_frames // 2
        )
        if not 0 <= self.anchor_frame < self.window_frames:
            raise ValueError("anchor_frame must be inside the temporal window")
        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        dim = self.encoder.out_dim
        temporal_ffn_dim = int(encoder_kwargs.get("ffn_dim", dim * 4))
        self.temporal_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.window_frames + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=int(encoder_kwargs.get("num_heads", 6)),
            dim_feedforward=temporal_ffn_dim, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(
            layer, num_layers=int(temporal_depth), enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_gate = nn.Parameter(torch.tensor(float(temporal_gate_init)))
        self.root_head = PoseHead(dim, dropout_p)
        self.relative_head = PoseHead(dim, dropout_p)
        nn.init.trunc_normal_(self.temporal_query, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def load_temporal_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = {
            "temporal_query", "temporal_pos", "temporal_norm.weight", "temporal_norm.bias",
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if name in expected or name.startswith("temporal_blocks.")]
        unexpected = list(unexpected)
        if missing or unexpected:
            raise RuntimeError(f"temporal downstream checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.window_frames:
            raise ValueError(
                f"expected temporal CSI (B,{self.window_frames},3,114,10), got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        frame_cls = self.encoder(x.reshape(batch * self.window_frames, *x.shape[2:]))
        frame_cls = frame_cls.view(batch, self.window_frames, -1)
        sequence = torch.cat([
            self.temporal_query.expand(batch, -1, -1), frame_cls,
        ], dim=1) + self.temporal_pos
        temporal_cls = self.temporal_norm(self.temporal_blocks(sequence))[:, 0]
        anchor_cls = frame_cls[:, self.anchor_frame]
        gate = torch.sigmoid(self.temporal_gate)
        fused = anchor_cls + gate * (temporal_cls - anchor_cls)

        # Absolute translation is read from the anchor frame.  The temporal
        # branch predicts only root-relative structure, preventing temporal
        # averaging from erasing scene-dependent global position.
        root = self.root_head(anchor_cls)[:, :1]
        relative = self.relative_head(fused)
        relative = relative - relative[:, :1]
        return root + relative


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


class SharedCLSStructuredPoseModel(nn.Module):
    """Downstream model that reads the pretrained shared CLS latent."""

    def __init__(self, dropout_p: float = 0.3, **encoder_kwargs):
        super().__init__()
        encoder_kwargs.pop("predictor_depth", None)
        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        self.shared_projection = nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        self.encoder_readout = "shared_cls"
        self.head = PoseHead(self.encoder.out_dim, dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        cls = self.encoder(x)
        return self.head(self.shared_projection(cls))


class SharedFusionStructuredPoseModel(nn.Module):
    """Fuse raw CLS with the pretrained shared CLS using a learned gate."""

    def __init__(self, dropout_p: float = 0.3, **encoder_kwargs):
        super().__init__()
        encoder_kwargs.pop("predictor_depth", None)
        self.encoder = StructuredViTEncoder(**encoder_kwargs)
        self.shared_projection = nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        self.shared_norm = nn.LayerNorm(self.encoder.out_dim)
        self.shared_gate = nn.Parameter(torch.tensor(-1.1))
        self.encoder_readout = "shared_fusion"
        self.head = PoseHead(self.encoder.out_dim, dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            x = x.squeeze(1)
        raw_cls = self.encoder(x)
        shared_cls = self.shared_norm(self.shared_projection(raw_cls))
        gate = torch.sigmoid(self.shared_gate)
        fused = raw_cls + gate * (shared_cls - raw_cls)
        return self.head(fused)
