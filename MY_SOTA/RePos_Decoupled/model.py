from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Use the official MM-Fi H36M joint layout from the main RePos implementation.
# The old 343-mm copy carried a stale arm topology (it attached the two
# wrists to different shoulder branches), which made the pose tokenizer and
# limb losses supervise the wrong body graph.
from MY_SOTA.RePos.model import ASPN, BONES, PARTS, PART_COUNTS
from MY_SOTA.RePos.model import ChebGraphConv, SkeletonGraphAttention
from WIFIJEPA.src.masking import make_link_mask
from WIFIJEPA.src.model import Predictor, StructuredViTEncoder

_JOINT_PART = [0] * 17
for _part, _joints in enumerate(PARTS):
    for _joint in _joints:
        _JOINT_PART[_joint] = _part
JOINT_PART = tuple(_JOINT_PART)


def root_relative(pose: Tensor) -> Tensor:
    return pose - pose[..., :1, :]


class CSIMotionTokenizer(nn.Module):
    """Full CSI tokenizer with structured MAE + link-level JEPA pretraining."""

    def __init__(self, embed_dim=256, depth=4, num_heads=8, ffn_dim=1024,
                 predictor_depth=2, n_time=10, n_links=3, n_subcarriers=114,
                 mask_ratio=0.75, n_masked_links=1, decoder_dim=None,
                 decoder_depth=2, decoder_heads=None):
        super().__init__()
        self.n_time, self.n_links, self.n_subcarriers = int(n_time), int(n_links), int(n_subcarriers)
        self.n_tokens = self.n_time * self.n_links
        self.mask_ratio, self.n_masked_links = float(mask_ratio), int(n_masked_links)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be strictly between 0 and 1")
        self.context_encoder = StructuredViTEncoder(embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, ffn_dim=ffn_dim, n_time=n_time, n_links=n_links)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = Predictor(embed_dim=embed_dim, depth=predictor_depth,
            num_heads=num_heads, n_tokens=self.n_tokens)
        decoder_dim = int(decoder_dim or embed_dim)
        decoder_heads = int(decoder_heads or num_heads)
        if decoder_dim % decoder_heads:
            raise ValueError("decoder_dim must be divisible by decoder_heads")
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_mask = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.n_tokens, decoder_dim))
        layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=decoder_depth, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, n_subcarriers)
        nn.init.trunc_normal_(self.decoder_mask, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _check_input(self, x):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        expected = (self.n_links, self.n_subcarriers, self.n_time)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected CSI shape (B,{expected[0]},{expected[1]},{expected[2]})")
        return x

    def _raw_targets(self, x):
        target = x.permute(0, 3, 1, 2).reshape(x.shape[0], self.n_tokens, self.n_subcarriers)
        mean = target.mean(dim=-1, keepdim=True)
        std = target.var(dim=-1, keepdim=True, unbiased=True).add(1e-6).sqrt()
        return (target - mean) / std

    def encode(self, x, *, return_cls=False):
        x = self._check_input(x)
        output = self.context_encoder(x, return_tokens=True)
        cls, tokens = output[:, 0], output[:, 1:]
        return (tokens, cls) if return_cls else tokens

    def forward(self, x):
        return self.encode(x)

    @staticmethod
    def _augment(x, noise_std=0.01):
        shifts = torch.randint(-1, 2, (x.shape[0],), device=x.device)
        shifted = torch.stack([torch.roll(sample, int(shift), dims=-1)
                               for sample, shift in zip(x, shifts)], dim=0)
        return shifted + torch.randn_like(shifted) * float(noise_std)

    def _mae_loss(self, x, ids_keep, ids_restore):
        tokens = self.context_encoder.tokenizer(x)
        gather = ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        visible = torch.gather(tokens, 1, gather)
        encoded = self.context_encoder._encode(visible)
        decoded_visible = self.decoder_embed(encoded[:, 1:])
        batch, n_keep = ids_keep.shape
        mask_tokens = self.decoder_mask.expand(batch, self.n_tokens - n_keep, -1)
        patches = torch.cat([decoded_visible, mask_tokens], dim=1)
        restore = ids_restore.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        patches = torch.gather(patches, 1, restore) + self.decoder_pos
        predicted = self.decoder_pred(self.decoder_norm(self.decoder(patches)))
        target = self._raw_targets(x)
        mask = torch.ones(batch, self.n_tokens, device=x.device)
        mask[:, :n_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        token_loss = (predicted - target).pow(2).mean(dim=-1)
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)

    def pretrain_loss(self, x, *, noise_std=0.01, neighbor=None,
                      neighbor_available=None, sequence_ids=None, temporal_weight=0.0):
        x = self._check_input(x)
        batch = x.shape[0]
        n_keep = max(1, int(round(self.n_tokens * (1.0 - self.mask_ratio))))
        ids_shuffle = torch.argsort(torch.rand(batch, self.n_tokens, device=x.device), dim=1)
        ids_keep, ids_restore = ids_shuffle[:, :n_keep], torch.argsort(ids_shuffle, dim=1)
        mae = self._mae_loss(x, ids_keep, ids_restore)
        target_mask = make_link_mask(batch, self.n_time, self.n_links, self.n_masked_links, x.device)
        cls, visible, visible_ids = self.context_encoder.forward_visible(x, ~target_mask)
        predicted = self.predictor(cls, visible, visible_ids)
        with torch.no_grad():
            target = F.layer_norm(self.target_encoder(x, return_tokens=True)[:, 1:], (self.context_encoder.out_dim,))
        jepa = F.smooth_l1_loss(predicted[target_mask], target[target_mask])
        _, aug_cls = self.encode(self._augment(x, noise_std=noise_std), return_cls=True)
        consistency = 1.0 - F.cosine_similarity(cls, aug_cls, dim=-1).mean()
        temporal = x.new_zeros(())
        if neighbor is not None and temporal_weight:
            valid = neighbor_available.bool() if neighbor_available is not None else torch.ones(batch, dtype=torch.bool, device=x.device)
            if valid.sum() > 1:
                _, anchor_cls = self.encode(x[valid], return_cls=True)
                with torch.no_grad():
                    next_cls = self.target_encoder(neighbor[valid], return_tokens=True)[:, 0]
                logits = F.normalize(anchor_cls.float(), dim=-1) @ F.normalize(next_cls.float(), dim=-1).T / 0.1
                eligible = torch.ones(logits.shape[0], dtype=torch.bool, device=x.device)
                if sequence_ids is not None:
                    ids = sequence_ids[valid]
                    same_sequence = ids[:, None].eq(ids[None, :])
                    eligible = (~same_sequence).any(dim=1)
                    same_sequence.fill_diagonal_(False)
                    logits = logits.masked_fill(same_sequence, torch.finfo(logits.dtype).min)
                if eligible.any():
                    temporal = F.cross_entropy(logits[eligible], torch.arange(logits.shape[0], device=x.device)[eligible])
        loss = mae + jepa + 0.25 * consistency + float(temporal_weight) * temporal
        return loss, {"loss": float(loss.detach()), "mae": float(mae.detach()),
                      "jepa": float(jepa.detach()), "temporal_consistency": float(consistency.detach()),
                      "adjacent_contrast": float(temporal.detach()),
                      "masked_tokens": float(self.n_tokens - n_keep)}

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(context.data, alpha=1.0 - momentum)


class PoseStructureTokenizer(nn.Module):
    """Full pose-only tokenizer with bone graph refinement and temporal support."""

    def __init__(self, embed_dim=256, depth=4, num_heads=8, ffn_dim=1024,
                 n_joints=17, gcn_layers=3, joint_attention_layers=2, mask_ratio=0.3):
        super().__init__()
        self.n_joints, self.mask_ratio = int(n_joints), float(mask_ratio)
        self.input_projection = nn.Linear(3, embed_dim)
        self.joint_pos = nn.Parameter(torch.zeros(1, n_joints, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, 512, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.gcn = nn.ModuleList([ChebGraphConv(embed_dim, order=3) for _ in range(gcn_layers)])
        self.joint_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                dim_feedforward=ffn_dim, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(joint_attention_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pose_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 3))
        self.register_buffer("joint_adjacency", self._make_joint_adjacency(), persistent=False)
        nn.init.trunc_normal_(self.joint_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _make_joint_adjacency():
        adjacency = torch.eye(17)
        for left, right in BONES:
            adjacency[left, right] = 1
            adjacency[right, left] = 1
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _check_input(self, pose):
        if pose.ndim == 4 and pose.shape[1] == 1:
            pose = pose.squeeze(1)
        if pose.ndim not in {3, 4} or tuple(pose.shape[-2:]) != (self.n_joints, 3):
            raise ValueError(f"expected pose shape (B,{self.n_joints},3) or (B,T,{self.n_joints},3)")
        return pose

    def _encode_frame(self, pose):
        tokens = self.input_projection(root_relative(pose)) + self.joint_pos
        tokens = self.norm(self.blocks(tokens))
        for index, layer in enumerate(self.gcn):
            tokens = tokens + F.gelu(layer(tokens, self.joint_adjacency))
            if index < len(self.joint_attention):
                tokens = self.joint_attention[index](tokens)
        return self.norm(tokens)

    def encode(self, pose):
        pose = self._check_input(pose)
        if pose.ndim == 3:
            return self._encode_frame(pose)
        batch, steps = pose.shape[:2]
        frame = self._encode_frame(pose.reshape(batch * steps, self.n_joints, 3)).reshape(batch, steps, self.n_joints, -1)
        temporal = frame.permute(0, 2, 1, 3).reshape(batch * self.n_joints, steps, -1)
        temporal = self.norm(self.blocks(temporal + self.temporal_pos[:, :steps]))
        return temporal[:, -1].reshape(batch, self.n_joints, -1)

    def decode(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.n_joints:
            raise ValueError("pose latent must have shape (B,n_joints,embed_dim)")
        return root_relative(self.pose_head(tokens))

    def forward(self, pose):
        return self.decode(self.encode(pose))

    def pretrain_loss(self, pose):
        pose = self._check_input(pose)
        target = root_relative(pose[:, -1] if pose.ndim == 4 else pose)
        batch = target.shape[0]
        n_mask = max(1, int(round(self.n_joints * self.mask_ratio)))
        raw = self.input_projection(target) + self.joint_pos
        ids = torch.argsort(torch.rand(batch, self.n_joints, device=target.device), dim=1)[:, :n_mask]
        masked = raw.clone()
        masked_pos = self.joint_pos.expand(batch, -1, -1).gather(1, ids.unsqueeze(-1).expand(-1, -1, raw.shape[-1]))
        masked.scatter_(1, ids.unsqueeze(-1).expand(-1, -1, raw.shape[-1]), self.mask_token.expand(batch, n_mask, -1) + masked_pos)
        encoded = self.norm(self.blocks(masked))
        for index, layer in enumerate(self.gcn):
            encoded = encoded + F.gelu(layer(encoded, self.joint_adjacency))
            if index < len(self.joint_attention):
                encoded = self.joint_attention[index](encoded)
        predicted = self.decode(self.norm(encoded))
        masked_pred = predicted.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
        masked_target = target.gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
        reconstruction = F.smooth_l1_loss(masked_pred, masked_target)
        bone = torch.stack([F.smooth_l1_loss(predicted[:, left] - predicted[:, right], target[:, left] - target[:, right]) for left, right in BONES]).mean()
        full_predicted = self.decode(self._encode_frame(target))
        full_reconstruction = F.smooth_l1_loss(full_predicted, target)
        velocity = predicted.new_zeros(())
        if pose.ndim == 4 and pose.shape[1] > 1:
            batch, steps = pose.shape[:2]
            frame_latents = self._encode_frame(pose.reshape(batch * steps, self.n_joints, 3))
            frame_predicted = self.decode(frame_latents).reshape(batch, steps, self.n_joints, 3)
            frame_target = root_relative(pose)
            velocity = F.smooth_l1_loss(
                frame_predicted[:, 1:] - frame_predicted[:, :-1],
                frame_target[:, 1:] - frame_target[:, :-1],
            )
        loss = reconstruction + full_reconstruction + 0.25 * bone + 0.1 * velocity
        return loss, {"loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()), "full_reconstruction": float(full_reconstruction.detach()), "bone": float(bone.detach()), "velocity": float(velocity.detach()), "masked_joints": float(n_mask)}


class FullBodyTranslator(nn.Module):
    """Complete RePos BP-LQ, SGA, attention pooling, GCN and joint decoder."""

    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, query_depth=1,
                 sga_depth=4, gcn_layers=3, joint_attention_layers=2, dropout=0.0,
                 translator_depth=1):
        super().__init__()
        self.part_queries = nn.Parameter(torch.randn(1, 6, embed_dim) * 0.02)
        self.part_expanders = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, count * embed_dim), nn.GELU())
            for count in PART_COUNTS
        ])
        self.part_projection = nn.Linear(embed_dim, embed_dim)
        self.part_pos = nn.ParameterList([nn.Parameter(torch.randn(1, count, embed_dim) * 0.02) for count in PART_COUNTS])
        cross_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.part_cross = nn.TransformerDecoder(cross_layer, num_layers=translator_depth)
        self.sga = SkeletonGraphAttention(embed_dim, heads=num_heads, depth=sga_depth, dropout=dropout)
        self.joint_queries = nn.Parameter(torch.randn(1, 17, embed_dim) * 0.02)
        self.joint_part_embed = nn.Parameter(torch.randn(6, embed_dim) * 0.02)
        joint_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.joint_decoder = nn.TransformerDecoder(joint_layer, num_layers=query_depth)
        self.joint_norm = nn.LayerNorm(embed_dim)
        self.gcn = nn.ModuleList([ChebGraphConv(embed_dim, order=3) for _ in range(gcn_layers)])
        self.joint_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                dim_feedforward=ffn_dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(joint_attention_layers)
        ])
        self.pose_head = nn.Linear(embed_dim, 3)
        self.register_buffer("joint_adjacency", self._make_joint_adjacency(), persistent=False)

    @staticmethod
    def _make_joint_adjacency():
        adjacency = torch.eye(17)
        for left, right in BONES:
            adjacency[left, right] = 1
            adjacency[right, left] = 1
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

    @staticmethod
    def _part_ranges():
        ranges, start = [], 0
        for count in PART_COUNTS:
            ranges.append((start, start + count))
            start += count
        return tuple(ranges)

    def forward(self, csi_tokens):
        if csi_tokens.ndim != 3:
            raise ValueError("CSI tokens must have shape (B,n_tokens,embed_dim)")
        batch = csi_tokens.shape[0]
        queries = self.part_queries.expand(batch, -1, -1)
        parts = self.part_cross(queries, csi_tokens)
        expanded = []
        for index, (expand, pos) in enumerate(zip(self.part_expanders, self.part_pos, strict=True)):
            count = PART_COUNTS[index]
            expanded.append(self.part_projection(expand(parts[:, index]).view(batch, count, -1)) + pos)
        body_tokens = self.sga(torch.cat(expanded, dim=1))
        pool_queries = self.part_queries.expand(batch, -1, -1)
        part_features = []
        for index, (start, end) in enumerate(self._part_ranges()):
            tokens = body_tokens[:, start:end]
            scores = (tokens * pool_queries[:, index:index + 1]).sum(dim=-1) / math.sqrt(tokens.shape[-1])
            part_features.append((tokens * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1))
        part_features = torch.stack(part_features, dim=1)
        part_ids = torch.as_tensor(JOINT_PART, device=csi_tokens.device)
        joint_queries = self.joint_queries.expand(batch, -1, -1) + self.joint_part_embed[part_ids][None]
        decoded = self.joint_norm(self.joint_decoder(joint_queries, part_features))
        for index, layer in enumerate(self.gcn):
            decoded = decoded + F.gelu(layer(decoded, self.joint_adjacency))
            if index < len(self.joint_attention):
                decoded = self.joint_attention[index](decoded)
        decoded = self.joint_norm(decoded)
        return root_relative(self.pose_head(decoded)), decoded


class DecoupledRePos(nn.Module):
    """Complete decoupled RePos: CSI tokenizer, pose tokenizer, translator, ASPN."""

    def __init__(self, *, embed_dim=256, depth=4, num_heads=8, ffn_dim=1024,
                 predictor_depth=2, n_time=10, n_links=3, n_subcarriers=114,
                 translator_depth=1, sga_depth=4, gcn_layers=3,
                 joint_attention_layers=2, query_depth=1, n_angle=90,
                 phase_mode="paper", freeze_tokenizers=True, pose_decode="tokenizer",
                 delay_mode="legacy_truncate"):
        super().__init__()
        if not freeze_tokenizers:
            raise ValueError("This adaptation protocol requires frozen tokenizers")
        if pose_decode not in {"tokenizer", "direct"}:
            raise ValueError("pose_decode must be tokenizer or direct")
        self.pose_decode = pose_decode
        self.csi_tokenizer = CSIMotionTokenizer(embed_dim, depth, num_heads, ffn_dim,
            predictor_depth, n_time, n_links, n_subcarriers)
        self.pose_tokenizer = PoseStructureTokenizer(embed_dim, depth, num_heads, ffn_dim,
            gcn_layers=gcn_layers, joint_attention_layers=joint_attention_layers)
        self.translator = FullBodyTranslator(embed_dim, num_heads, ffn_dim, query_depth,
            sga_depth=sga_depth, gcn_layers=gcn_layers, joint_attention_layers=joint_attention_layers,
            translator_depth=translator_depth)
        self.aspn = ASPN(n_time=n_time, n_links=n_links, n_subcarriers=n_subcarriers,
            n_angle=n_angle, phase_mode=phase_mode, delay_mode=delay_mode)
        if freeze_tokenizers:
            self.freeze_tokenizers()

    def freeze_tokenizers(self):
        for module in (self.csi_tokenizer, self.pose_tokenizer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward_relative(self, x, *, return_latent=False):
        pose, latent = self.translator(self.csi_tokenizer.encode(x))
        if self.pose_decode == "tokenizer":
            pose = self.pose_tokenizer.decode(latent)
        return (pose, latent) if return_latent else pose

    def forward_root(self, x):
        return self.aspn(x)

    def forward_absolute(self, x):
        return self.forward_relative(x) + self.forward_root(x)[:, None, :]

    def adaptation_loss(self, x, target_pose, *, latent_weight=0.1, limb_weight=0.1):
        target_relative = root_relative(target_pose)
        predicted, predicted_latent = self.forward_relative(x, return_latent=True)
        with torch.no_grad():
            target_latent = self.pose_tokenizer.encode(target_relative)
        pose_loss = F.l1_loss(predicted, target_relative) + F.mse_loss(predicted, target_relative)
        limb_loss = torch.stack([F.smooth_l1_loss(predicted[:, left] - predicted[:, right], target_relative[:, left] - target_relative[:, right]) for left, right in BONES]).mean()
        latent_loss = F.smooth_l1_loss(predicted_latent, target_latent)
        loss = pose_loss + float(limb_weight) * limb_loss + float(latent_weight) * latent_loss
        return loss, {"loss": float(loss.detach()), "pose": float(pose_loss.detach()), "limb": float(limb_loss.detach()), "latent": float(latent_loss.detach())}


def build_decoupled_from_config(config: dict[str, Any]) -> DecoupledRePos:
    model, inp = config.get("model", config), config.get("input", {})
    return DecoupledRePos(embed_dim=int(model.get("embed_dim", 256)), depth=int(model.get("depth", 4)),
        num_heads=int(model.get("num_heads", 8)), ffn_dim=int(model.get("ffn_dim", 1024)),
        predictor_depth=int(model.get("predictor_depth", 2)), n_time=int(inp.get("time", 10)),
        n_links=int(inp.get("links", 3)), n_subcarriers=int(inp.get("subcarriers", 114)),
        translator_depth=int(model.get("translator_depth", 1)), sga_depth=int(model.get("sga_depth", 4)),
        gcn_layers=int(model.get("gcn_layers", 3)), joint_attention_layers=int(model.get("joint_attention_layers", 2)),
        query_depth=int(model.get("query_depth", 1)), n_angle=int(model.get("n_angle", 90)),
        phase_mode=str(model.get("phase_mode", "paper")), freeze_tokenizers=bool(model.get("freeze_tokenizers", True)),
        pose_decode=str(model.get("pose_decode", "tokenizer")), delay_mode=str(model.get("delay_mode", "legacy_truncate")))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DecoupledRePos(embed_dim=64, depth=1, num_heads=8, ffn_dim=128, predictor_depth=1)
    x, y = torch.randn(2, 3, 114, 10), torch.randn(2, 17, 3)
    rel = model.forward_relative(x)
    loss, logs = model.adaptation_loss(x, y)
    ssl_csi, _ = model.csi_tokenizer.pretrain_loss(x)
    ssl_pose, _ = model.pose_tokenizer.pretrain_loss(y)
    assert rel.shape == (2, 17, 3)
    assert all(torch.isfinite(value) for value in (loss, ssl_csi, ssl_pose))
    print(tuple(rel.shape), float(loss.detach()), logs)
