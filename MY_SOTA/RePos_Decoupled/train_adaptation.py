from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmfi_wifi.metrics import evaluate_pose_official, mpjpe_mm, pa_mpjpe_mm
from scripts.metafi_ssl.train_supervised import load_audited_artifacts
from WIFIJEPA.src.data import WiFiJEPAData, _load_ground_truth, csi_preprocessing
from WIFIJEPA.src.paths import audit_paths, repo_path
from MY_SOTA.RePos_Decoupled.model import DecoupledRePos, build_decoupled_from_config


class PoseFrameDataset(Dataset):
    def __init__(self, dataset_root: str, keys, *, protocol: str, split: str, scope: str, manifest_fingerprint: str, normalize: bool):
        self.base = WiFiJEPAData(dataset_root, keys, neighbor_offsets=(), protocol=protocol, split=split, scope=scope, manifest_fingerprint=manifest_fingerprint, normalize=normalize)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        record = self.base._records[index]
        x = self.base[index].anchor
        gt_path = self.base.dataset_root / record.sequence_key.scene / record.sequence_key.subject / record.sequence_key.action / "ground_truth.npy"
        y = torch.as_tensor(_load_ground_truth(str(gt_path))[record.frame_index].copy(), dtype=torch.float32)
        return x, y


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: DecoupledRePos, loader: DataLoader, device: torch.device, *, absolute: bool, use_amp: bool) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model.forward_absolute(x) if absolute else model.forward_relative(x)
            predictions.append(pred.float().cpu().numpy())
            targets.append(y.float().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    if absolute:
        absolute_mpjpe, pa_mpjpe = evaluate_pose_official(pred, target)
        pelvis = mpjpe_mm(pred, target)
    else:
        target_relative = target - target[:, :1]
        absolute_mpjpe = float("nan")
        pelvis = mpjpe_mm(pred, target_relative, already_root_relative=True)
        pa_mpjpe = pa_mpjpe_mm(pred, target_relative, already_root_relative=True)
    root_error = float(np.linalg.norm(pred[:, 0] - target[:, 0], axis=-1).mean() * 1000.0) if absolute else float('nan')
    return {"absolute_mpjpe_mm": float(absolute_mpjpe), "pelvis_mpjpe_mm": float(pelvis), "pa_mpjpe_mm": float(pa_mpjpe), "root_mpjpe_mm": root_error}


def load_pretrained(model: DecoupledRePos, config: dict, manifest, labels) -> None:
    pretrain = config.get("pretrain", {})
    csi_path = pretrain.get("csi_checkpoint")
    pose_path = pretrain.get("pose_checkpoint")
    if csi_path:
        checkpoint = torch.load(repo_path(csi_path), map_location="cpu", weights_only=True)
        if checkpoint.get('kind') != 'csi_tokenizer' or checkpoint.get('data_source') != 'audited_mmfi' or checkpoint.get('manifest_fingerprint') != manifest.fingerprint():
            raise ValueError('CSI tokenizer checkpoint does not match the audited MM-Fi manifest')
        model.csi_tokenizer.load_state_dict(checkpoint["state_dict"], strict=True)
    if pose_path:
        checkpoint = torch.load(repo_path(pose_path), map_location="cpu", weights_only=True)
        if checkpoint.get('kind') != 'pose_tokenizer' or checkpoint.get('data_source') != 'audited_mmfi' or checkpoint.get('manifest_fingerprint') != manifest.fingerprint() or checkpoint.get('label_manifest_fingerprint') != labels.fingerprint or checkpoint.get('pose_layout') != 'mmfi_3d_h36m_root0_v1' or checkpoint.get('topology_version') != 2 or checkpoint.get('objective_version') != 3:
            raise ValueError('Pose tokenizer checkpoint does not match the audited MM-Fi 4-shot labels')
        model.pose_tokenizer.load_state_dict(checkpoint["state_dict"], strict=True)


def run(config: dict, eval_checkpoint: str | None = None) -> Path:
    exp, ft = config["experiment"], config["finetune"]
    seed_all(int(exp["seed"]))
    device_name = str(exp.get("device", "cuda"))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    use_amp = bool(ft.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    normalize = not bool(config["input"].get("disable_normalization", False))
    audit = audit_paths(config)
    manifest, labels, _ = load_audited_artifacts(audit["manifest"], audit["label_manifest"], audit["leakage_audit"], audit["axis_stats"], protocol=exp["protocol"], split=exp["split"], label_budget=exp["label_budget"])
    dataset_root = repo_path(exp["dataset_root"])
    make_ds = lambda keys: PoseFrameDataset(str(dataset_root), keys, protocol=manifest.protocol, split=manifest.split, scope=manifest.scope, manifest_fingerprint=manifest.fingerprint(), normalize=normalize)
    train_ds, select_ds, test_ds = make_ds(labels.selected_keys), make_ds(manifest.select_keys), make_ds(manifest.test_keys)
    train_loader = DataLoader(train_ds, batch_size=int(ft.get("batch_size", 64)), shuffle=True, drop_last=True, num_workers=0)
    select_loader = DataLoader(select_ds, batch_size=int(ft.get("eval_batch_size", 128)), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(ft.get("eval_batch_size", 128)), shuffle=False, num_workers=0)
    model = build_decoupled_from_config(config).to(device)
    pretrain_cfg = config.get("pretrain", {})
    if bool(config.get("model", {}).get("freeze_tokenizers", True)) and not (pretrain_cfg.get("csi_checkpoint") and pretrain_cfg.get("pose_checkpoint")):
        raise ValueError("freeze_tokenizers=true requires both pretrain.csi_checkpoint and pretrain.pose_checkpoint")
    if eval_checkpoint is None:
        load_pretrained(model, config, manifest, labels)
    output = repo_path(exp["output_root"]) / exp["protocol"] / exp["split"] / f"seed{int(exp['seed'])}" / exp["label_budget"]
    output.mkdir(parents=True, exist_ok=True)
    if eval_checkpoint is None and (any(output.glob('*.pth')) or (output / 'done.txt').exists()):
        raise FileExistsError(f'Existing training results in {output}; choose a new output_root')
    if eval_checkpoint is not None:
        checkpoint_path = repo_path(eval_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get('stage') != 'root':
            raise ValueError('Evaluation requires a root-stage checkpoint')
        if checkpoint.get('architecture_version') != 3:
            raise ValueError('Checkpoint uses a different architecture; evaluate it with the original model code')
        saved_config = checkpoint.get('config')
        if not isinstance(saved_config, dict) or any(saved_config.get(key) != config.get(key) for key in ('experiment', 'input', 'model', 'pretrain', 'finetune')):
            raise ValueError('Evaluation config does not match the checkpoint config')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        with csi_preprocessing(normalize):
            test = evaluate(model, test_loader, device, absolute=True, use_amp=use_amp)
        result = {'method': 'repos_decoupled', 'checkpoint': str(checkpoint_path),
                  'checkpoint_metrics': checkpoint.get('metrics'), 'test_metrics': test,
                  'note': 'Evaluation of a saved checkpoint; training history is unavailable.'}
        (output / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f"[EVAL] epoch={checkpoint['metrics']['epoch']} test_abs={test['absolute_mpjpe_mm']:.2f}mm test_pelvis={test['pelvis_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm", flush=True)
        return output
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    history = {"translator": [], "root": []}
    translator_checkpoint = None
    root_checkpoint = None

    # Stage 1 learns only the cross-modal translation interface.
    optimizer = torch.optim.AdamW(model.translator.parameters(), lr=float(ft.get("translator_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
    best = float("inf")
    with csi_preprocessing(normalize):
        for epoch in range(int(ft.get("translator_epochs", 8))):
            model.train(); model.csi_tokenizer.eval(); model.pose_tokenizer.eval(); totals = {}
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, logs = model.adaptation_loss(x, y, latent_weight=float(ft.get("latent_weight", 0.1)), limb_weight=float(ft.get("limb_weight", 0.1)))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.translator.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                for key, value in logs.items(): totals[key] = totals.get(key, 0.0) + value
            metrics = evaluate(model, select_loader, device, absolute=False, use_amp=use_amp)
            row = {"epoch": epoch + 1, **{f"train_{key}": value / max(1, len(train_loader)) for key, value in totals.items()}, **metrics}
            history["translator"].append(row)
            print(f"[translator {epoch + 1}] loss={row['train_loss']:.5f} select_rel={row['pelvis_mpjpe_mm']:.2f}mm", flush=True)
            if metrics["pelvis_mpjpe_mm"] < best:
                best = metrics["pelvis_mpjpe_mm"]
                translator_checkpoint = output / f"translator_epoch_{epoch + 1:03d}.pth"
                if translator_checkpoint.exists():
                    raise FileExistsError(f"Checkpoint exists; choose a new output directory: {translator_checkpoint}")
                torch.save({"stage": "translator", "topology_version": 2, "architecture_version": 3, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, translator_checkpoint)
        model.load_state_dict(torch.load(translator_checkpoint, map_location=device, weights_only=True)["model_state_dict"], strict=True)

        # Stage 2 keeps the decoupled relative branch fixed and trains ASPN root.
        for parameter in model.csi_tokenizer.parameters(): parameter.requires_grad_(False)
        for parameter in model.pose_tokenizer.parameters(): parameter.requires_grad_(False)
        for parameter in model.translator.parameters(): parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.aspn.parameters(), lr=float(ft.get("root_lr", 3e-4)), weight_decay=float(ft.get("weight_decay", 1e-4)))
        best = float("inf")
        for epoch in range(int(ft.get("root_epochs", 10))):
            model.train(); model.csi_tokenizer.eval(); model.pose_tokenizer.eval(); model.translator.eval(); total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(): relative = model.forward_relative(x)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    root = model.forward_root(x)
                    absolute = relative.detach() + root[:, None, :]
                    loss = torch.nn.functional.l1_loss(absolute, y) + torch.nn.functional.mse_loss(absolute, y) + float(ft.get("root_weight", 1.0)) * torch.nn.functional.l1_loss(root, y[:, 0])
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.aspn.parameters(), 1.0); scaler.step(optimizer); scaler.update(); total += float(loss.detach())
            metrics = evaluate(model, select_loader, device, absolute=True, use_amp=use_amp)
            row = {"epoch": epoch + 1, "train_loss": total / max(1, len(train_loader)), **metrics}
            history["root"].append(row)
            print(f"[root {epoch + 1}] loss={row['train_loss']:.5f} select_abs={row['absolute_mpjpe_mm']:.2f}mm", flush=True)
            if metrics["absolute_mpjpe_mm"] < best:
                best = metrics["absolute_mpjpe_mm"]
                root_checkpoint = output / f"root_epoch_{epoch + 1:03d}.pth"
                if root_checkpoint.exists():
                    raise FileExistsError(f"Checkpoint exists; choose a new output directory: {root_checkpoint}")
                torch.save({"stage": "root", "topology_version": 2, "architecture_version": 3, "model_state_dict": model.state_dict(), "metrics": row, "config": config}, root_checkpoint)
        model.load_state_dict(torch.load(root_checkpoint, map_location=device, weights_only=True)["model_state_dict"], strict=True)
        test = evaluate(model, test_loader, device, absolute=True, use_amp=use_amp)
    (output / "metrics.json").write_text(json.dumps({"method": "repos_decoupled", "test_metrics": test, "history": history}, indent=2), encoding="utf-8")
    (output / "done.txt").write_text("complete\n", encoding="utf-8")
    print(f"[DONE] test_abs={test['absolute_mpjpe_mm']:.2f}mm test_pelvis={test['pelvis_mpjpe_mm']:.2f}mm test_pa={test['pa_mpjpe_mm']:.2f}mm", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="MY_SOTA/RePos_Decoupled/config.yaml")
    parser.add_argument("--eval-checkpoint", default=None, help="Evaluate an existing root-stage checkpoint without training")
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")), eval_checkpoint=args.eval_checkpoint)


if __name__ == "__main__":
    main()
