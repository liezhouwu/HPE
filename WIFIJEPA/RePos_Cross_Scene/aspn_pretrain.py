from __future__ import annotations

import argparse
import json
import random
import sys
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


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(samples):
    return torch.stack([sample.anchor for sample in samples])


class ASPNPretrain(nn.Module):
    model_variant = "repos_aspn_amplitude_pretrain_v1"

    def __init__(self, n_time=10, n_links=3, n_subcarriers=114, n_angle=90):
        super().__init__()
        self.aspn = ASPN(n_time=n_time, n_links=n_links, n_subcarriers=n_subcarriers, n_angle=n_angle, phase_mode="paper")
        self.stats_head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, n_links * 4))
        self.n_links = n_links

    def stats(self, x):
        flat = x.float().reshape(x.shape[0], self.n_links, -1)
        return torch.stack((flat.mean(-1), flat.std(-1, unbiased=False), flat.square().mean(-1).sqrt(), flat.abs().amax(-1)), dim=-1)

    def forward(self, x):
        # Masking one physical link teaches the ASPN stem to use the remaining propagation paths.
        masked = x.clone(); link = torch.randint(self.n_links, (x.shape[0],), device=x.device)
        masked[torch.arange(x.shape[0], device=x.device), link] = 0
        features = self.aspn.forward_features(masked)
        pred = self.stats_head(features).view(x.shape[0], self.n_links, 4)
        target = self.stats(x)
        stats_loss = F.smooth_l1_loss(pred, target)
        clean = self.aspn.forward_features(x)
        jitter = self.aspn.forward_features(x * (0.95 + 0.1 * torch.rand(x.shape[0], 1, 1, 1, device=x.device)))
        consistency_loss = F.smooth_l1_loss(F.normalize(clean, dim=-1), F.normalize(jitter, dim=-1))
        return stats_loss + 0.2 * consistency_loss, {"loss": float((stats_loss + 0.2 * consistency_loss).detach()), "stats_loss": float(stats_loss.detach()), "consistency_loss": float(consistency_loss.detach())}

    def export_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.aspn.state_dict().items()}


def run(config):
    run_cfg, inp, pre = config["experiment"], config["input"], config["pretrain"]
    seed = int(run_cfg["seed"]); seed_all(seed)
    audit = audit_paths(config); full = DataManifest.read_json(audit["manifest"])
    manifest = make_small_pretrain_manifest(full, fraction=float(pre.get("sequence_fraction", 0.2)), seed=seed)
    root = repo_path(run_cfg["dataset_root"]); normalize = not bool(inp.get("disable_normalization", False))
    ds = WiFiJEPAData(str(root), manifest.pretrain_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    loader = DataLoader(ds, batch_size=int(pre.get("batch_size", 64)), shuffle=True, drop_last=True, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    device = torch.device(str(run_cfg.get("device", "cuda")) if str(run_cfg.get("device", "cuda")) == "cpu" or torch.cuda.is_available() else "cpu")
    model = ASPNPretrain(n_time=int(inp["time"]), n_links=int(inp["links"]), n_subcarriers=int(inp["subcarriers"]), n_angle=90).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(pre.get("learning_rate", 3e-4)), weight_decay=float(pre.get("weight_decay", 1e-4)))
    amp = bool(pre.get("amp", True)) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = repo_path(run_cfg["output_root"]); out.mkdir(parents=True, exist_ok=True); (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = []
    with csi_preprocessing(normalize):
        for epoch in range(int(pre.get("epochs", 10))):
            sums = {"loss": 0.0, "stats_loss": 0.0, "consistency_loss": 0.0}
            for x in loader:
                x = x.to(device); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp): loss, log = model(x)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                for k in sums: sums[k] += log[k]
            row = {k: v / max(1, len(loader)) for k, v in sums.items()}; row["epoch"] = epoch + 1; metrics.append(row); print(f"[aspn {epoch + 1}] loss={row['loss']:.5f} stats={row['stats_loss']:.5f}", flush=True)
    ckpt = {"method": "repos_aspn_pretrain", "model_variant": ASPNPretrain.model_variant, "status": "complete", "seed": seed, "manifest_fingerprint": manifest.fingerprint(), "config": config, "aspn_state_dict": model.export_state_dict(), "metrics": metrics}
    torch.save(ckpt, out / "aspn.pth"); (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(f"[DONE] {out / 'aspn.pth'}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="WIFIJEPA/RePos_Cross_Scene/config_aspn_pretrain.yaml"); a = p.parse_args(); run(yaml.safe_load(Path(a.config).read_text(encoding="utf-8-sig")))


if __name__ == "__main__": main()
