from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.data_manifest import DataManifest
from scripts.metafi_ssl.train_supervised import load_audited_artifacts
from WIFIJEPA.src.data import WiFiJEPAData, _load_ground_truth, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, repo_path
from WIFIJEPA.RePos_Cross_Scene.model import ASPN


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class RootAwareDataset(Dataset):
    """Labeled source-scene frames plus a same-sequence temporal partner."""

    def __init__(self, dataset_root, keys, *, protocol, split, scope, manifest_fingerprint, normalize):
        self.base = WiFiJEPAData(dataset_root, keys, protocol=protocol, split=split, scope=scope, manifest_fingerprint=manifest_fingerprint, normalize=normalize)
        self.root = Path(dataset_root)

    def __len__(self): return len(self.base)

    def _pose(self, record):
        path = self.root / record.sequence_key.scene / record.sequence_key.subject / record.sequence_key.action / "ground_truth.npy"
        return torch.as_tensor(_load_ground_truth(str(path))[record.frame_index].copy(), dtype=torch.float32)

    def __getitem__(self, index):
        record = self.base._records[index]
        partner_index = record.frame_index + 1 if record.frame_index < 296 else record.frame_index - 1
        partner_path = record.frame_path.with_name(f"frame{partner_index + 1:03d}.mat")
        partner = type(record)(record.sequence_key, partner_index, record.packed_path, partner_path)
        x = self.base[index].anchor
        # WiFiJEPAData exposes the frame index through a record; using a tiny
        # direct loader avoids crossing sequence boundaries.
        partner_sample = self.base._load_record(partner)
        return x, self._pose(record), partner_sample, self._pose(partner)


def collate(items):
    return tuple(torch.stack([item[i] for item in items]) for i in range(4))


def run(config):
    exp, inp, pre = config["experiment"], config["input"], config["pretrain"]
    seed = int(exp["seed"]); seed_all(seed)
    audit = audit_paths(config)
    manifest, labels, _ = load_audited_artifacts(audit["manifest"], audit["label_manifest"], audit["leakage_audit"], audit["axis_stats"], protocol=exp["protocol"], split=exp["split"], label_budget=exp["label_budget"])
    dataset_root = repo_path(exp["dataset_root"]); normalize = not bool(inp.get("disable_normalization", False))
    dataset = RootAwareDataset(str(dataset_root), labels.selected_keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    loader = DataLoader(dataset, batch_size=int(pre.get("batch_size", 64)), shuffle=True, drop_last=True, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    device_name = str(exp.get("device", "cuda")); device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = ASPN(n_time=int(inp["time"]), n_links=int(inp["links"]), n_subcarriers=int(inp["subcarriers"]), n_angle=90, phase_mode="paper").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pre.get("learning_rate", 3e-4)), weight_decay=float(pre.get("weight_decay", 1e-4)))
    epochs = int(pre.get("epochs", 10)); amp = bool(pre.get("amp", True)) and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = repo_path(exp["output_root"]); output.mkdir(parents=True, exist_ok=True); (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics = []
    print(f"[root-aware pretrain] device={device} frames={len(dataset)} batches={len(loader)} output={output}", flush=True)
    with csi_preprocessing(normalize):
        for epoch in range(epochs):
            totals = {"loss": 0.0, "root": 0.0, "delta": 0.0, "aug": 0.0}; started = time.perf_counter(); model.train()
            for x, y, x2, y2 in loader:
                x, y, x2, y2 = x.to(device), y.to(device), x2.to(device), y2.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    root1 = model(x); root2 = model(x2)
                    root_loss = F.smooth_l1_loss(root1, y[:, 0]) + F.smooth_l1_loss(root2, y2[:, 0])
                    delta_loss = F.smooth_l1_loss(root2 - root1, y2[:, 0] - y[:, 0])
                    gain = 0.95 + 0.1 * torch.rand(x.shape[0], 1, 1, 1, device=device)
                    aug_root = model(x * gain + torch.randn_like(x) * 0.005)
                    aug_loss = F.smooth_l1_loss(aug_root, root1.detach())
                    loss = root_loss + float(pre.get("lambda_delta", 0.5)) * delta_loss + float(pre.get("lambda_aug", 0.1)) * aug_loss
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                for k, v in (("loss", loss), ("root", root_loss), ("delta", delta_loss), ("aug", aug_loss)): totals[k] += float(v.detach())
            row = {k: v / max(1, len(loader)) for k, v in totals.items()}; row.update(epoch=epoch + 1, seconds=time.perf_counter() - started); metrics.append(row)
            print(f"[root-aware {epoch + 1:02d}/{epochs:02d}] loss={row['loss']:.5f} root={row['root']:.5f} delta={row['delta']:.5f} aug={row['aug']:.5f}", flush=True)
    checkpoint = {"method": "repos_root_aware_pretrain", "model_variant": "repos_aspn_root_aware_v1", "status": "complete", "seed": seed, "manifest_fingerprint": manifest.fingerprint(), "label_budget": exp["label_budget"], "config": config, "aspn_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "metrics": metrics}
    path = output / "aspn_root_aware.pth"; torch.save(checkpoint, path); (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(f"[DONE] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="WIFIJEPA/RePos_Cross_Scene/config_root_aware_pretrain.yaml"); args = parser.parse_args(); run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))


if __name__ == "__main__": main()
