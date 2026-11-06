from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest
from WIFIJEPA.src.data import WiFiJEPAData, csi_preprocessing
from WIFIJEPA.src.model import StructuredViTEncoder
from WIFIJEPA.src.paths import audit_paths, repo_path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(samples):
    return torch.stack([sample.anchor for sample in samples])


class AdaptedRePosPretrain(nn.Module):
    """Pretraining objective aligned with RePos token and root branches."""

    model_variant = "repos_adapted_body_residual_v1"

    def __init__(self, *, embed_dim=256, depth=4, num_heads=8, ffn_dim=1024,
                 n_time=10, n_links=3, mask_links=1, stats_dim=4,
                 lambda_body=1.0, lambda_consistency=0.5,
                 lambda_residual=0.5, lambda_stats=0.2):
        super().__init__()
        kwargs = dict(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                      ffn_dim=ffn_dim, n_time=n_time, n_links=n_links)
        self.encoder = StructuredViTEncoder(**kwargs)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.residual_predictor = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.stats_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, n_links * stats_dim))
        self.n_time, self.n_links, self.mask_links = n_time, n_links, mask_links
        self.stats_dim = stats_dim
        self.lambda_body = float(lambda_body)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_residual = float(lambda_residual)
        self.lambda_stats = float(lambda_stats)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float):
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - momentum)

    def _mask(self, batch: int, device: torch.device) -> Tensor:
        # Same physical links are masked for every time token in a sample.
        scores = torch.rand(batch, self.n_links, device=device)
        links = scores.argsort(dim=1)[:, :self.mask_links]
        link_mask = torch.zeros(batch, self.n_links, dtype=torch.bool, device=device)
        link_mask.scatter_(1, links, True)
        return link_mask[:, None, :].expand(batch, self.n_time, self.n_links).reshape(batch, -1)

    def _stats(self, x: Tensor) -> Tensor:
        # Preserve per-link scale cues for the ASPN/root branch.
        flat = x.float().permute(0, 1, 2, 3).reshape(x.shape[0], self.n_links, -1)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1, unbiased=False)
        rms = flat.square().mean(dim=-1).sqrt()
        peak = flat.abs().amax(dim=-1)
        return torch.stack((mean, std, rms, peak), dim=-1)

    def forward(self, x: Tensor):
        batch = x.shape[0]
        mask = self._mask(batch, x.device)
        tokens = self.encoder.tokenizer(x)
        masked_tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        online = self.encoder._encode(masked_tokens)
        with torch.no_grad():
            target = self.target_encoder(x, return_tokens=True)
            target_tokens = target[:, 1:]
        pred = self.predictor(online[:, 1:])
        target_norm = F.layer_norm(target_tokens, (target_tokens.shape[-1],))
        body_loss = F.smooth_l1_loss(pred[mask], target_norm[mask])

        online_links = online[:, 1:].view(batch, self.n_time, self.n_links, -1)
        target_links = target_tokens.view(batch, self.n_time, self.n_links, -1)
        online_res = online_links - online_links.mean(dim=2, keepdim=True)
        target_res = target_links - target_links.mean(dim=2, keepdim=True)
        residual_pred = self.residual_predictor(online_links.reshape(batch, -1, online_links.shape[-1])).view_as(online_links)
        residual_loss = F.smooth_l1_loss(residual_pred[mask.view(batch, self.n_time, self.n_links)], target_res[mask.view(batch, self.n_time, self.n_links)])

        body_global = F.normalize(online_links.mean(dim=(1, 2)), dim=-1)
        full_global = F.normalize(target_links.mean(dim=(1, 2)), dim=-1)
        consistency_loss = (1.0 - (body_global * full_global).sum(dim=-1)).mean()

        stats_target = self._stats(x)
        stats_pred = self.stats_head(online[:, 0]).view(batch, self.n_links, self.stats_dim)
        stats_loss = F.smooth_l1_loss(stats_pred, stats_target)
        loss = (self.lambda_body * body_loss + self.lambda_consistency * consistency_loss
                + self.lambda_residual * residual_loss + self.lambda_stats * stats_loss)
        return loss, {"loss": float(loss.detach()), "body_loss": float(body_loss.detach()),
                      "consistency_loss": float(consistency_loss.detach()),
                      "residual_loss": float(residual_loss.detach()), "stats_loss": float(stats_loss.detach())}

    def export_encoder_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}


def run(config: dict) -> Path:
    run_cfg, input_cfg, model_cfg, pre_cfg = config["experiment"], config["input"], config["model"], config["pretrain"]
    seed = int(run_cfg["seed"])
    seed_all(seed)
    audit = audit_paths(config)
    full_manifest = DataManifest.read_json(audit["manifest"])
    manifest = make_small_pretrain_manifest(full_manifest, fraction=float(pre_cfg.get("sequence_fraction", 0.2)), seed=seed)
    root = repo_path(run_cfg["dataset_root"])
    normalize = not bool(input_cfg.get("disable_normalization", False))
    dataset = WiFiJEPAData(str(root), manifest.pretrain_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    loader = DataLoader(dataset, batch_size=int(pre_cfg.get("batch_size", 64)), shuffle=True, drop_last=True, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    device = torch.device(str(run_cfg.get("device", "cuda")) if str(run_cfg.get("device", "cuda")) == "cpu" or torch.cuda.is_available() else "cpu")
    model = AdaptedRePosPretrain(embed_dim=int(model_cfg["embed_dim"]), depth=int(model_cfg["depth"]), num_heads=int(model_cfg["num_heads"]), ffn_dim=int(model_cfg["ffn_dim"]), n_time=int(input_cfg["time"]), n_links=int(input_cfg["links"]), mask_links=int(pre_cfg.get("mask_links", 1)), lambda_body=float(pre_cfg.get("lambda_body", 1.0)), lambda_consistency=float(pre_cfg.get("lambda_consistency", 0.5)), lambda_residual=float(pre_cfg.get("lambda_residual", 0.5)), lambda_stats=float(pre_cfg.get("lambda_stats", 0.2))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pre_cfg.get("learning_rate", 3e-4)), weight_decay=float(pre_cfg.get("weight_decay", 1e-4)))
    epochs = int(pre_cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(pre_cfg.get("lr_min", 1e-5)))
    amp = bool(pre_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = repo_path(run_cfg["output_root"]); output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = []
    print(f"[adapted pretrain] device={device} frames={len(dataset)} batches={len(loader)} output={output}", flush=True)
    with csi_preprocessing(normalize):
        for epoch in range(epochs):
            model.train(); totals = {k: 0.0 for k in ("loss", "body_loss", "consistency_loss", "residual_loss", "stats_loss")}; started = time.perf_counter()
            for x in loader:
                x = x.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp): loss, log = model(x)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                for key in totals: totals[key] += log[key]
                model.update_target(float(pre_cfg.get("ema", 0.996)))
            scheduler.step(); row = {k: v / max(1, len(loader)) for k, v in totals.items()}; row.update(epoch=epoch + 1, seconds=time.perf_counter() - started); metrics.append(row)
            print(f"[epoch {epoch + 1:02d}/{epochs:02d}] loss={row['loss']:.5f} body={row['body_loss']:.5f} residual={row['residual_loss']:.5f} stats={row['stats_loss']:.5f}", flush=True)
    checkpoint = {"method": "repos_adapted_pretrain", "model_variant": AdaptedRePosPretrain.model_variant, "status": "complete", "seed": seed, "manifest_fingerprint": manifest.fingerprint(), "config": config, "encoder_config": {"embed_dim": int(model_cfg["embed_dim"]), "depth": int(model_cfg["depth"]), "num_heads": int(model_cfg["num_heads"]), "ffn_dim": int(model_cfg["ffn_dim"]), "n_time": int(input_cfg["time"]), "n_links": int(input_cfg["links"])}, "encoder_state_dict": model.export_encoder_state_dict(), "metrics": metrics}
    path = output / "encoder.pth"; torch.save(checkpoint, path); (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(f"[DONE] {path}", flush=True); return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="WIFIJEPA/RePos_Cross_Scene/config_pretrain_adapted.yaml"); args = parser.parse_args(); run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))


if __name__ == "__main__":
    main()
