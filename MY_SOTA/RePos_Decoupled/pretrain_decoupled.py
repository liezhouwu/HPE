from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MY_SOTA.RePos_Decoupled.model import CSIMotionTokenizer, PoseStructureTokenizer
from WIFIJEPA.src.data import WiFiJEPAData
from WIFIJEPA.src.paths import audit_paths
from scripts.metafi_ssl.train_supervised import load_audited_artifacts
from pose_ssl.metafi.pretrain_subset import sample_pretrain_sequences
from pose_ssl.metafi.label_stats import fingerprint_keys


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value).resolve()


def _audited_data(section: dict):
    audit_config = yaml.safe_load(_resolve(section['audit_config']).read_text(encoding='utf-8-sig'))
    exp = audit_config['experiment']
    paths = audit_paths(audit_config)
    manifest, labels, _ = load_audited_artifacts(
        paths['manifest'], paths['label_manifest'], paths['leakage_audit'], paths['axis_stats'],
        protocol=exp['protocol'], split=exp['split'], label_budget=exp['label_budget'],
    )
    if manifest.scope != 'strict':
        raise ValueError('Pretraining requires a strict manifest')
    return audit_config, manifest, labels


def _csi_dataset(section: dict):
    audit_config, manifest, labels = _audited_data(section)
    fraction = float(section['dataset_fraction'])
    if not 0 < fraction <= 1:
        raise ValueError('dataset_fraction must be in (0, 1]')
    required = frozenset(labels.selected_keys)
    pool = frozenset(manifest.pretrain_keys)
    if not required.issubset(pool):
        raise ValueError('4-shot sequences must be part of the allowed pretraining pool')
    total_sequences = len(manifest.official_train_keys | manifest.test_keys)
    target_count = round(total_sequences * fraction)
    if target_count < len(required) or target_count > len(pool):
        raise ValueError(f'{target_count} sequences cannot contain {len(required)} 4-shot sequences within the {len(pool)}-sequence pretraining pool')
    remaining = pool - required
    extra_count = target_count - len(required)
    if extra_count:
        selected = sample_pretrain_sequences(remaining, fraction=extra_count / len(remaining), seed=manifest.seed, split=manifest.split)
        generator = torch.Generator().manual_seed(manifest.seed)
        if len(selected) > extra_count:
            ordered = sorted(selected)
            selected = frozenset(ordered[i] for i in torch.randperm(len(ordered), generator=generator)[:extra_count].tolist())
        elif len(selected) < extra_count:
            ordered = sorted(remaining - selected)
            selected = selected | frozenset(ordered[i] for i in torch.randperm(len(ordered), generator=generator)[:extra_count - len(selected)].tolist())
    else:
        selected = frozenset()
    selected = required | selected
    dataset = WiFiJEPAData(
        str(_resolve(section['dataset_root'])), selected, neighbor_offsets=(1,),
        protocol=manifest.protocol, split=manifest.split, scope=manifest.scope,
        manifest_fingerprint=manifest.fingerprint(),
        normalize=not bool(audit_config['input'].get('disable_normalization', False)),
    )
    return dataset, manifest.fingerprint()


def _pose_dataset(section: dict):
    _, manifest, labels = _audited_data(section)
    if not labels.selected_keys or not labels.selected_keys.issubset(manifest.internal_train_keys):
        raise ValueError('Pose pretraining labels must be confined to internal training sequences')
    if labels.selected_keys & (manifest.select_keys | manifest.test_keys):
        raise ValueError('Pose pretraining labels overlap select or test sequences')
    root = _resolve(section['dataset_root'])
    poses = []
    for key in sorted(labels.selected_keys):
        path = root / key.scene / key.subject / key.action / 'ground_truth.npy'
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[1:] != (17, 3) or not np.isfinite(array).all():
            raise ValueError(f'Invalid MM-Fi pose array: {path}, shape={array.shape}')
        poses.append(array.astype(np.float32, copy=False))
    values = torch.from_numpy(np.concatenate(poses, axis=0))
    return TensorDataset(values), manifest.fingerprint(), labels.fingerprint, len(labels.selected_keys)


def _collate_temporal(samples):
    keys = {key: index for index, key in enumerate(dict.fromkeys(sample.sequence_key for sample in samples))}
    anchors = torch.stack([sample.anchor for sample in samples])
    neighbors = torch.stack([sample.neighbors[0] if sample.neighbors else torch.zeros_like(sample.anchor) for sample in samples])
    available = torch.tensor([bool(sample.neighbors) for sample in samples], dtype=torch.bool)
    sequence_ids = torch.tensor([keys[sample.sequence_key] for sample in samples], dtype=torch.long)
    return anchors, neighbors, available, sequence_ids


def _run(mode: str, output: Path, epochs: int, batch_size: int, device: str, *, section: dict, seed: int) -> None:
    output = _resolve(output)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if section.get('source') != 'audited_mmfi':
        raise ValueError('Only audited MM-Fi data is permitted')
    if output.exists():
        raise FileExistsError(f'Checkpoint exists; choose a new output path: {output}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    audit_config, _, _ = _audited_data(section)
    model_config = audit_config['model']
    input_config = audit_config['input']
    common_model = dict(embed_dim=int(model_config['embed_dim']), depth=int(model_config['depth']),
                        num_heads=int(model_config['num_heads']), ffn_dim=int(model_config['ffn_dim']))
    label_fingerprint = None
    if mode == 'csi':
        dataset, manifest_fingerprint = _csi_dataset(section)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=_collate_temporal)
        model = CSIMotionTokenizer(**common_model, predictor_depth=int(model_config.get('predictor_depth', 2)),
                                   n_time=int(input_config['time']), n_links=int(input_config['links']),
                                   n_subcarriers=int(input_config['subcarriers'])).to(torch_device)
        input_shape = (int(input_config['links']), int(input_config['subcarriers']), int(input_config['time']))
        source = str(_resolve(section['dataset_root']))
        print(f"[csi] sequences={len(dataset.sequence_keys)} frames={len(dataset)} subset={fingerprint_keys(dataset.sequence_keys)} manifest={manifest_fingerprint}", flush=True)
    else:
        dataset, manifest_fingerprint, label_fingerprint, sequence_count = _pose_dataset(section)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        input_shape = (17, 3)
        source = str(_resolve(section['dataset_root']))
        model = PoseStructureTokenizer(**common_model, gcn_layers=int(model_config.get('gcn_layers', 3)),
                                       joint_attention_layers=int(model_config.get('joint_attention_layers', 2))).to(torch_device)
        print(f"[pose] 4-shot sequences={sequence_count} frames={len(dataset)} manifest={manifest_fingerprint} labels={label_fingerprint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        totals = {}
        for item in loader:
            batch = item[0]
            batch = batch.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            if mode == 'csi':
                neighbor, available, sequence_ids = (value.to(torch_device) for value in item[1:])
                loss, logs = model.pretrain_loss(batch, neighbor=neighbor, neighbor_available=available,
                                                sequence_ids=sequence_ids, temporal_weight=float(section.get('temporal_weight', 0.0)))
            else:
                loss, logs = model.pretrain_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if mode == 'csi':
                model.update_target()
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        row = {"epoch": epoch + 1, **{key: value / max(1, len(loader)) for key, value in totals.items()}}
        history.append(row)
        print(f"[{mode}] epoch={epoch + 1:03d}/{epochs:03d} loss={row['loss']:.6f}", flush=True)
    checkpoint = {
        "kind": f"{mode}_tokenizer",
        "state_dict": model.state_dict(),
        "input_shape": input_shape,
        "history": history,
        "source": source,
        "manifest_fingerprint": manifest_fingerprint,
        "label_manifest_fingerprint": label_fingerprint,
        "data_source": "audited_mmfi",
        "seed": seed,
    }
    if mode == 'pose':
        checkpoint['pose_layout'] = 'mmfi_3d_h36m_root0_v1'
        checkpoint['topology_version'] = 2
        checkpoint['objective_version'] = 3
    if mode == 'csi':
        checkpoint['dataset_fraction'] = float(section['dataset_fraction'])
        checkpoint['pretrain_sequence_fingerprint'] = fingerprint_keys(dataset.sequence_keys)
        checkpoint['pretrain_sequence_keys'] = [[k.scene, k.subject, k.action] for k in dataset.sequence_keys]
    torch.save(checkpoint, output)
    output.with_suffix(".json").write_text(json.dumps(checkpoint | {"state_dict": None}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose-free CSI or pose-only tokenizer pretraining")
    parser.add_argument("mode", nargs="?", choices=("csi", "pose"))
    parser.add_argument("--config", type=Path, default=Path('MY_SOTA/RePos_Decoupled/pretrain.yaml'), help="MM-Fi pretraining YAML config")
    args = parser.parse_args()
    config = yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8-sig"))
    common = config['pretrain']
    modes = (args.mode,) if args.mode else ('csi', 'pose')
    for mode in modes:
        section = common[mode]
        if section.get('source') != 'audited_mmfi':
            raise ValueError('Only audited MM-Fi data is permitted')
        _audited_data(section)
        if _resolve(section['output']).exists():
            raise FileExistsError(f"Checkpoint exists; choose a new output path: {_resolve(section['output'])}")
    for mode in modes:
        section = common[mode]
        _run(mode, Path(section['output']), int(common['epochs']), int(common['batch_size']), str(common['device']), section=section, seed=int(common.get('seed', 42)))


if __name__ == "__main__":
    main()
