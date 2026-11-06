from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from pose_ssl.metafi.pretrain_subset import make_small_pretrain_manifest
from WIFIJEPA.src.data import WiFiJEPAData, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, repo_path
from WIFIJEPA.RePos_Cross_Scene.model import ASPN


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(samples):
    return torch.stack([sample.anchor for sample in samples])


class ASPNGeometricPretrain(nn.Module):
    model_variant = "repos_aspn_geometric_phase_v2"

    def __init__(self, n_time=10, n_links=3, n_subcarriers=114, n_angle=90, n_delay=64):
        super().__init__()
        self.online = ASPN(n_time=n_time, n_links=n_links, n_subcarriers=n_subcarriers, n_angle=n_angle, n_delay=n_delay, phase_mode="paper")
        self.teacher = copy.deepcopy(self.online)
        for p in self.teacher.parameters(): p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode); self.teacher.eval(); return self

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        for online, teacher in zip(self.online.parameters(), self.teacher.parameters(), strict=True):
            teacher.lerp_(online, 1.0 - momentum)

    @staticmethod
    def normalize_map(value):
        value = torch.log1p(value.clamp_min(0))
        return (value - value.mean(dim=(2, 3), keepdim=True)) / value.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)

    def forward(self, x):
        batch = x.shape[0]
        with torch.no_grad():
            target = self.normalize_map(self.teacher.forward_heatmap(x))
        masked = x.clone()
        links = torch.randint(self.online.n_links, (batch,), device=x.device)
        masked[torch.arange(batch, device=x.device), links] = 0
        gain = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=x.device)
        noisy = masked * gain + torch.randn_like(masked) * 0.01
        prediction = self.normalize_map(self.online.forward_heatmap(noisy))
        masked_loss = F.smooth_l1_loss(prediction, target)

        view = x * gain
        consistency = F.smooth_l1_loss(
            self.normalize_map(self.online.forward_heatmap(view)), target
        )
        # Encourage the phase branch to remain bounded and avoid arbitrary phase explosions.
        phase_reg = self.online.phase_net[-2].weight.square().mean()
        loss = masked_loss + 0.5 * consistency + 1e-3 * phase_reg
        return loss, {"loss": float(loss.detach()), "masked_heatmap": float(masked_loss.detach()), "consistency": float(consistency.detach()), "phase_reg": float(phase_reg.detach())}

    def export_phase_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.online.phase_net.state_dict().items()}


def run(config):
    run_cfg, inp, pre = config["experiment"], config["input"], config["pretrain"]
    seed = int(run_cfg["seed"]); seed_all(seed)
    audit = audit_paths(config); full = DataManifest.read_json(audit["manifest"])
    manifest = make_small_pretrain_manifest(full, fraction=float(pre.get("sequence_fraction", 0.2)), seed=seed)
    root = repo_path(run_cfg["dataset_root"]); normalize = not bool(inp.get("disable_normalization", False))
    ds = WiFiJEPAData(str(root), manifest.pretrain_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    loader = DataLoader(ds, batch_size=int(pre.get("batch_size", 64)), shuffle=True, drop_last=True, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    device = torch.device(str(run_cfg.get("device", "cuda")) if str(run_cfg.get("device", "cuda")) == "cpu" or torch.cuda.is_available() else "cpu")
    model = ASPNGeometricPretrain(n_time=int(inp["time"]), n_links=int(inp["links"]), n_subcarriers=int(inp["subcarriers"]), n_angle=90).to(device)
    opt = torch.optim.AdamW(model.online.phase_net.parameters(), lr=float(pre.get("learning_rate", 1e-4)), weight_decay=float(pre.get("weight_decay", 1e-5)))
    amp = bool(pre.get("amp", True)) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs = int(pre.get("epochs", 10)); out = repo_path(run_cfg["output_root"]); out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = []
    with csi_preprocessing(normalize):
        for epoch in range(epochs):
            sums = {"loss": 0.0, "masked_heatmap": 0.0, "consistency": 0.0, "phase_reg": 0.0}; started = time.perf_counter()
            for x in loader:
                x = x.to(device); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp): loss, log = model(x)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.online.phase_net.parameters(), 1.0); scaler.step(opt); scaler.update(); model.update_teacher(float(pre.get("ema", 0.996)))
                for k in sums: sums[k] += log[k]
            row = {k: v / max(1, len(loader)) for k, v in sums.items()}; row.update(epoch=epoch + 1, seconds=time.perf_counter() - started); metrics.append(row)
            print(f"[geometric {epoch + 1:02d}/{epochs:02d}] loss={row['loss']:.5f} masked={row['masked_heatmap']:.5f} consistency={row['consistency']:.5f}", flush=True)
    ckpt = {"method": "repos_aspn_geometric_pretrain", "model_variant": ASPNGeometricPretrain.model_variant, "status": "complete", "seed": seed, "manifest_fingerprint": manifest.fingerprint(), "config": config, "phase_net_state_dict": model.export_phase_state_dict(), "metrics": metrics}
    path = out / "phase_net.pth"; torch.save(ckpt, path); (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(f"[DONE] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="WIFIJEPA/RePos_Cross_Scene/config_aspn_geometric_pretrain.yaml"); args = parser.parse_args(); run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))


if __name__ == "__main__": main()
