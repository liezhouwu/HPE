from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mmfi_wifi.data import _load_ground_truth
from mmfi_wifi.data_manifest import DataManifest, audit_manifest
from mmfi_wifi.sequence_keys import SequenceKey
from MY_SOTA.RadioInversePose.src.model import RadioInversePose, observation


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_frame(path):
    import scipy.io as sio
    amp_file = path.with_suffix(".npy")
    phase_file = path.with_suffix(".phase.npy")
    payload = sio.loadmat(path, variable_names=["CSIphase"] if amp_file.is_file() else ["CSIamp", "CSIphase"]) if not phase_file.is_file() or not amp_file.is_file() else {}
    amp = np.load(amp_file) if amp_file.is_file() else payload["CSIamp"]
    phase = np.load(phase_file) if phase_file.is_file() else payload["CSIphase"]
    result = []
    for x in (amp, phase):
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (3, 114, 10): raise ValueError(f"unexpected CSI shape: {x.shape} in {path}")
        valid = np.isfinite(x)
        fill = float(x[valid].mean()) if valid.any() else 0.0
        result.append(torch.from_numpy(np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)))
    return result


class Frames(Dataset):
    def __init__(self, root, keys, stride=1):
        self.root = root
        self.keys = sorted(keys)
        self.items = [(k, i) for k in self.keys for i in range(0, 297, stride)]
        if not self.items: raise ValueError("empty frame dataset")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        k, i = self.items[index]
        seq = self.root / k.scene / k.subject / k.action
        a, p = load_frame(seq / "wifi-csi" / f"frame{i + 1:03d}.mat")
        y = torch.from_numpy(np.array(_load_ground_truth(str(seq / "ground_truth.npy"))[i], dtype=np.float32, copy=True))
        return a, p, y, f"{k.scene}/{k.subject}/{k.action}"


def loader(dataset, cfg, shuffle=False):
    workers = int(cfg.get("num_workers", 0))
    args = {"batch_size": int(cfg.get("batch_size", 64)), "shuffle": shuffle, "num_workers": workers,
            "pin_memory": torch.cuda.is_available()}
    if workers: args.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **args)


def source_statistics(root, keys, cfg):
    obs, poses = [], []
    ds = Frames(root, keys, stride=int(cfg.get("statistics_stride", 30)))
    for a, p, y, _ in loader(ds, cfg):
        obs.append(observation(a, p)); poses.append(y.flatten(1))
    x, y = torch.cat(obs), torch.cat(poses)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError("nonfinite training statistics")
    mean = y.mean(0); _, _, vh = torch.linalg.svd(y - mean, full_matrices=False)
    rank = int(cfg.get("prior_rank", 16))
    if not 0 < rank <= min(vh.shape): raise ValueError("invalid pose prior rank")
    return x.mean(0), x.std(0, unbiased=False), mean, vh[:rank]


def pose_errors(p, y):
    p, y = p.reshape(-1, 17, 3), y.reshape(-1, 17, 3)
    absolute = (p - y).norm(dim=-1).mean(1) * 1000
    root = (p[:, 0] - y[:, 0]).norm(dim=-1) * 1000
    relative = ((p - p[:, :1]) - (y - y[:, :1])).norm(dim=-1).mean(1) * 1000
    pc, yc = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    u, s, vh = torch.linalg.svd(pc.transpose(1, 2) @ yc)
    d = torch.det(u @ vh).sign()
    correction = torch.eye(3, device=p.device).expand(len(p), -1, -1).clone(); correction[:, 2, 2] = d
    scale = (s * torch.stack((torch.ones_like(d), torch.ones_like(d), d), dim=1)).sum(1) / pc.square().sum((1, 2)).clamp_min(1e-8)
    aligned = scale[:, None, None] * (pc @ u @ correction @ vh) + y.mean(1, keepdim=True)
    pa = (aligned - y).norm(dim=-1).mean(1) * 1000
    return {"absolute_mpjpe_mm": absolute, "root_error_mm": root, "relative_mpjpe_mm": relative, "pa_mpjpe_mm": pa}


def evaluate(model, batches, device, inv, max_batches=0):
    model.eval(); sums = {}; seqs = {}; count = 0; started = time.perf_counter()
    for index, (a, p, target, keys) in enumerate(batches, 1):
        a, p, target = a.to(device), p.to(device), target.to(device)
        with torch.no_grad():
            obs = model.normalize(a, p); initial, z = model.initialize(obs)
            before_obj = model.objective(initial, z, obs, initial, inv["weights"])
            # Reverse order is a deterministic within-batch pose mismatch diagnostic.
            shuffled = model.render(target.flatten(1).flip(0), z)
            true = model.render(target.flatten(1), z)
            true_err = F.smooth_l1_loss(true, obs, reduction="none").mean(1)
            shuffled_err = F.smooth_l1_loss(shuffled, obs, reduction="none").mean(1)
        refined, _, after_obj = model.refine(obs.detach(), initial.detach(), z.detach(), steps=inv["steps"], learning_rate=inv["learning_rate"], weights=inv["weights"])
        with torch.no_grad():
            values = {"initial_" + k: v for k, v in pose_errors(initial, target).items()}
            values.update({"refined_" + k: v for k, v in pose_errors(refined, target).items()})
            values.update({"objective_initial": before_obj, "objective_refined": after_obj,
                           "true_pose_radio_error": true_err, "shuffled_pose_radio_error": shuffled_err,
                           "shuffled_is_different": (target.flatten(1) - target.flatten(1).flip(0)).norm(dim=1).gt(1e-6).float(),
                           "pose_update_mm": (refined - initial).reshape(-1, 17, 3).norm(dim=-1).mean(1) * 1000})
            for name, val in values.items(): sums[name] = sums.get(name, 0.) + float(val.sum())
            for j, key in enumerate(keys):
                row = seqs.setdefault(key, {"frames": 0, "initial_absolute_sum": 0., "refined_absolute_sum": 0.})
                row["frames"] += 1
                row["initial_absolute_sum"] += float(values["initial_absolute_mpjpe_mm"][j])
                row["refined_absolute_sum"] += float(values["refined_absolute_mpjpe_mm"][j])
        count += len(target)
        if max_batches and index >= max_batches: break
    if not count: raise ValueError("empty evaluation")
    result = {k: v / count for k, v in sums.items()}
    result.update(frames=count, seconds_per_frame=(time.perf_counter() - started) / count)
    for v in seqs.values():
        v["initial_absolute_mpjpe_mm"] = v.pop("initial_absolute_sum") / v["frames"]
        v["refined_absolute_mpjpe_mm"] = v.pop("refined_absolute_sum") / v["frames"]
    return result, seqs


def final_test(checkpoint, output, cfg, manifest, label_file, root, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state["mode"] != "final_source" or state["manifest_fingerprint"] != manifest.fingerprint()
            or state["label_sha256"] != hashlib.sha256(label_file.read_bytes()).hexdigest()):
        raise ValueError("checkpoint provenance mismatch")
    if state["config"] != cfg:
        raise ValueError("config differs from checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "final_report.json"
    if report_path.exists():
        raise FileExistsError(f"test report already exists: {report_path}")
    sd = state["model_state_dict"]
    model = RadioInversePose(sd["obs_mean"], sd["obs_std"], sd["pose_mean"], sd["pose_basis"], **cfg["model"]).to(device)
    model.load_state_dict(sd)
    print("[TEST] Evaluating the best source-validation checkpoint on all E04 test frames", flush=True)
    result, seqs = evaluate(model, loader(Frames(root, manifest.test_keys), cfg["experiment"]), device, cfg["inference"])
    report = {"method": "RadioInversePose", "protocol": manifest.protocol, "split": manifest.split,
              "scope": manifest.scope, "seed": manifest.seed, "label_budget": "4shot",
              "manifest_fingerprint": manifest.fingerprint(), "label_sha256": state["label_sha256"],
              "checkpoint": str(Path(checkpoint).resolve()), "best_epoch": state["epoch"],
              "select": state["select"], "test": result, "test_used_for_selection": False,
              "exploratory_e04": True,
              "test_independence": "E04 label statistics were inspected during prior method development"}
    (output / "test_per_sequence.json").write_text(json.dumps(seqs, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--holdout-scene", choices=["E01", "E02", "E03"])
    group.add_argument("--final-source-training", action="store_true", help="train all source labels and automatically test (default)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--test-checkpoint", type=Path)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); exp = cfg["experiment"]
    seed = int(exp["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manifest = DataManifest.read_json(resolve(exp["manifest"])); label_file = resolve(exp["label_manifest"])
    payload = json.loads(label_file.read_text(encoding="utf-8-sig")); keys = frozenset(SequenceKey(*v) for v in payload["selected_keys"])
    audit = audit_manifest(manifest, keys)
    if not audit.passed: raise ValueError(f"manifest audit failed: {audit.violations}")
    if (manifest.protocol, manifest.split, manifest.scope, manifest.seed) != ("protocol3", "cross_scene_split", "strict", seed): raise ValueError("unexpected protocol")
    if payload.get("mode") != "4shot": raise ValueError("expected 4shot manifest")
    root = resolve(exp["dataset_root"]); output = resolve(cfg["artifacts"]["output_root"])
    final_source = args.holdout_scene is None
    mode = "final_source" if final_source else f"holdout_{args.holdout_scene}"
    output = output / ("smoke_" + mode if args.smoke else mode)
    if args.test_checkpoint and (args.smoke or not final_source): raise ValueError("test requires final-source checkpoint and no smoke")
    train_keys = keys if final_source else frozenset(k for k in keys if k.scene != args.holdout_scene)
    select_keys = manifest.select_keys if final_source else frozenset(k for k in manifest.select_keys if k.scene == args.holdout_scene)
    if not train_keys or not select_keys or train_keys & select_keys: raise ValueError("invalid source split")
    device = torch.device(exp.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if args.test_checkpoint:
        final_test(args.test_checkpoint, output, cfg, manifest, label_file, root, device)
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pth").exists() and not args.smoke:
        raise FileExistsError(f"existing run in {output}; use a fresh output_root")
    stats = source_statistics(root, train_keys, exp)
    model = RadioInversePose(*stats, **cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(exp["learning_rate"]), weight_decay=float(exp["weight_decay"]))
    epochs = 1 if args.smoke else int(exp["epochs"]); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    train_loader = loader(Frames(root, train_keys), exp, True); select_loader = loader(Frames(root, select_keys), exp)
    print(json.dumps({"mode": mode, "train_sequences": len(train_keys), "select_sequences": len(select_keys), "device": str(device), "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    best = float("inf"); history = []; inv = cfg["inference"]
    for epoch in range(epochs):
        model.train(); total = 0.; n = 0
        for i, (a, p, target, _) in enumerate(train_loader, 1):
            a, p, target = a.to(device), p.to(device), target.to(device).flatten(1)
            obs = model.normalize(a, p); initial, z = model.initialize(obs)
            true_radio = model.render(target, z)
            radio_loss = F.smooth_l1_loss(true_radio, obs)
            refined, _, _ = model.refine(obs, initial, z, steps=int(cfg["training"]["unroll_steps"]), learning_rate=inv["learning_rate"], weights=inv["weights"], differentiable=True)
            loss = (F.smooth_l1_loss(initial, target) + F.smooth_l1_loss(refined, target)
                    + float(cfg["training"]["radio_weight"]) * radio_loss
                    + inv["weights"]["nuisance"] * z.square().mean())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            total += float(loss.detach()); n += 1
            interval = int(exp.get("log_interval", 100))
            if interval > 0 and i % interval == 0: print(f"epoch={epoch+1} batch={i} loss={total/n:.5f}", flush=True)
            if args.smoke and i >= 2: break
        scheduler.step(); val, seqs = evaluate(model, select_loader, device, inv, 2 if args.smoke else 0)
        row = {"epoch": epoch + 1, "train_loss": total / n, **val}; history.append(row); print(json.dumps(row), flush=True)
        if val["refined_absolute_mpjpe_mm"] < best:
            best = val["refined_absolute_mpjpe_mm"]
            torch.save({"mode": mode, "epoch": epoch + 1, "model_state_dict": model.state_dict(), "manifest_fingerprint": manifest.fingerprint(), "label_sha256": hashlib.sha256(label_file.read_bytes()).hexdigest(), "config": cfg, "select": row, "train_keys": [[k.scene,k.subject,k.action] for k in sorted(train_keys)]}, output / "best.pth")
            (output / "select_per_sequence.json").write_text(json.dumps(seqs, indent=2), encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if final_source and not args.smoke:
        final_test(output / "best.pth", output, cfg, manifest, label_file, root, device)
    print(f"completed: {output}", flush=True)


if __name__ == "__main__": main()
