"""
try3 — 测试 CLI
=================
按训练配置定义的 validation/test 划分评估模型。
口径由 config.target_space 决定:
  absolute      — 官方口径: 无对齐 MPJPE + PA-MPJPE (Table 3 对标), 另报骨盆对齐诊断值
  root_relative — 3D MPJPE (骨盆对齐) + PA-MPJPE
报告: 总体 / 逐关节 / 逐动作明细。

用法:
    python scripts/reproduction/test.py ../my_dataset [./config.yaml] --ckpt result/run_xxx/best_model.pth
"""

import os
import sys
import json
import hashlib
import argparse
import time
from collections import defaultdict

# UTF-8 patch
import builtins as _bi
_orig_open = _bi.open
def _utf8(file, mode='r', buffering=-1, encoding=None,
          errors=None, newline=None, closefd=True, opener=None):
    if encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
_bi.open = _utf8

import yaml
import numpy as np
import torch
import torch.nn as nn

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)

from mmfi_wifi.data import (make_dataset, make_dataloader, make_manual_test_dataset,
                            DAILY_ACTIONS, REHAB_ACTIONS)
from mmfi_wifi.engine import (setup_cuda, make_target, resolve_target_space,
                              config_fingerprint)
from mmfi_wifi.metrics import (evaluate_pose, evaluate_pose_official, mpjpe_mm,
                               per_joint_mpjpe_mm, KP_NAMES)
from mmfi_wifi.legacy_checkpoint import load_metafi_state_dict
from mmfi_wifi.model import posenet



def _eval_arrays(preds, gts, target_space):
    """返回 (mpjpe, pampjpe, mpjpe_pelvis)。preds/gts 与训练目标同坐标系。"""
    if target_space == 'absolute':
        m, pa = evaluate_pose_official(preds, gts)
        m_pelvis = mpjpe_mm(preds, gts, already_root_relative=False)
    else:
        m, pa = evaluate_pose(preds, gts, already_root_relative=False)
        m_pelvis = m
    return m, pa, m_pelvis


def _load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML mapping: {path}")
    for key in ('protocol', 'split_to_use', 'modality', 'data_unit'):
        if key not in config:
            raise ValueError(f"配置缺少必需字段 {key!r}: {path}")
    return config


def _config_fingerprint(config):
    """Return a stable SHA-256 fingerprint for JSON/YAML-compatible config data."""
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _select_config(explicit_path, ckpt_path):
    adjacent_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                 'config.yaml')
    explicit_config = _load_yaml_config(explicit_path) if explicit_path else None
    adjacent_config = _load_yaml_config(adjacent_path) if os.path.isfile(adjacent_path) else None

    if adjacent_config is not None:
        if explicit_config is not None and explicit_config != adjacent_config:
            raise ValueError(
                f"显式配置与 checkpoint 旁配置不一致: {explicit_path} != {adjacent_path}")
        return adjacent_config, adjacent_path, 'checkpoint_adjacent'
    if explicit_config is not None:
        return explicit_config, explicit_path, 'explicit'
    raise FileNotFoundError(
        f"未提供配置，且 checkpoint 旁不存在 config.yaml: {adjacent_path}")


def main():
    parser = argparse.ArgumentParser(description="try3 — 测试集评估")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("config_file", type=str, nargs='?', default=None,
                        help="配置文件；checkpoint 旁有 config.yaml 时优先使用并校验")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_cuda(device)
    use_amp = device.type == 'cuda'
    print(f"[INFO] Device: {device}", flush=True)

    config, config_path, config_source = _select_config(args.config_file, args.ckpt)
    config_path = os.path.abspath(config_path)
    print(f"[INFO] 配置: {config_path} ({config_source})", flush=True)
    print(f"[INFO] 评估身份: {config['protocol']} / {config['split_to_use']}", flush=True)

    # ---- 构建配置所定义的 held-out 测试集 ----
    if config['split_to_use'] == 'manual_split':
        test_ds = make_manual_test_dataset(args.dataset_root, config)
    else:
        _, test_ds = make_dataset(args.dataset_root, config)
    test_data_form = {
        subject: sorted(actions)
        for subject, actions in sorted(test_ds.data_source.items())
    }
    test_identity = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'dataset_split': test_ds.split,
        'subjects': sorted(test_data_form),
        'actions': sorted({action for actions in test_data_form.values()
                           for action in actions}),
        'data_form_fingerprint': _config_fingerprint(test_data_form),
    }
    rng = torch.Generator().manual_seed(config.get('init_rand_seed', 42))
    loader = make_dataloader(test_ds, False, rng, args.batch_size, args.num_workers)
    print(f"[INFO] 测试样本: {len(test_ds):,} ({len(loader)} batches)", flush=True)

    # ---- 模型 ----
    model = posenet(dropout_p=config.get('dropout_p', 0.0),
                    target_space=resolve_target_space(config)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    expected_fingerprint = ckpt.get('config_fingerprint')
    actual_fingerprint = config_fingerprint(config)
    if expected_fingerprint is not None and expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "checkpoint config_fingerprint 与评估配置不匹配: "
            f"{expected_fingerprint} != {actual_fingerprint}")
    # Legacy flat checkpoints and current nested checkpoints are both accepted.
    # Checkpoints before the output affine preserve the initialized affine values.
    missing, _ = load_metafi_state_dict(model, ckpt['model_state_dict'], strict=True)
    if missing:
        print(f"[INFO] checkpoint 无 {sorted(missing)} (旧版), "
              f"使用 target_space={resolve_target_space(config)} 的初始化值", flush=True)
    model.eval()
    print(f"[INFO] 模型已加载 (epoch {ckpt.get('epoch', '?')}, "
          f"ckpt metrics: {ckpt.get('metrics', {})})", flush=True)

    # ---- 评估 ----
    target_space = resolve_target_space(config)
    mpjpe_label = 'officialMPJPE' if target_space == 'absolute' else 'MPJPE'
    criterion = nn.MSELoss()
    all_preds, all_gts, all_actions = [], [], []
    total_loss = 0.0
    loss_samples = 0

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            csi = batch['input_wifi-csi'].unsqueeze(1).to(device, dtype=torch.float, non_blocking=True)
            gt = batch['output'].to(device, dtype=torch.float, non_blocking=True)
            target = make_target(gt, target_space)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(csi)
                loss = criterion(pred, target)
            total_loss += loss.item() * pred.shape[0]
            loss_samples += pred.shape[0]

            all_preds.append(pred.float().cpu().numpy())
            all_gts.append(target.float().cpu().numpy())
            actions = batch['action']
            if isinstance(actions, (list, tuple)):
                all_actions.extend(actions)
            else:
                all_actions.extend([actions] * pred.shape[0])

    eval_time = time.time() - t0
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    N = all_preds.shape[0]

    # 总体指标 (absolute: 官方无对齐; root_relative: 骨盆对齐)
    avg_mpjpe, avg_pampjpe, avg_mpjpe_pelvis = _eval_arrays(all_preds, all_gts, target_space)
    avg_loss = total_loss / max(loss_samples, 1)

    # 逐关节 (与总体同口径: absolute 时不做对齐)
    per_joint_rr = (target_space != 'absolute')
    if target_space == 'absolute':
        err = np.sqrt(np.sum(np.square(
            np.asarray(all_preds, dtype=np.float64) -
            np.asarray(all_gts, dtype=np.float64)), axis=2))
        per_joint = {KP_NAMES[j]: round(float(err[:, j].mean() * 1000.0), 2)
                     for j in range(17)}
    else:
        per_joint = {k: round(v, 2) for k, v in
                     per_joint_mpjpe_mm(all_preds, all_gts, already_root_relative=False).items()}

    # 逐动作
    per_action = defaultdict(lambda: {'preds': [], 'gts': []})
    for i, act in enumerate(all_actions):
        per_action[act]['preds'].append(all_preds[i])
        per_action[act]['gts'].append(all_gts[i])

    action_results = {}
    for act in sorted(per_action):
        ap = np.stack(per_action[act]['preds'], axis=0)
        ag = np.stack(per_action[act]['gts'], axis=0)
        m, pa, _ = _eval_arrays(ap, ag, target_space)
        cat = 'daily' if act in DAILY_ACTIONS else 'rehab'
        action_results[act] = {'category': cat, 'count': int(len(ap)),
                               'mpjpe_mm': round(m, 2), 'pampjpe_mm': round(pa, 2)}

    # 类别汇总 (daily / rehab)
    for cat_name, cat_acts in [('daily', DAILY_ACTIONS), ('rehab', REHAB_ACTIONS)]:
        cat_p, cat_g = [], []
        for act in cat_acts:
            if act in per_action:
                cat_p.extend(per_action[act]['preds'])
                cat_g.extend(per_action[act]['gts'])
        if cat_p:
            m, pa, _ = _eval_arrays(np.stack(cat_p), np.stack(cat_g), target_space)
            action_results[f'ALL_{cat_name}'] = {'category': cat_name, 'count': int(len(cat_p)),
                                                 'mpjpe_mm': round(m, 2), 'pampjpe_mm': round(pa, 2)}

    # ---- 打印 ----
    print(f"\n{'=' * 60}", flush=True)
    print(f"测试结果 ({N:,} 样本, {eval_time:.1f}s) | target_space={target_space}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Loss:       {avg_loss:.4f}")
    if target_space == 'absolute':
        print(f"  MPJPE:      {avg_mpjpe:.1f} mm   (官方口径: 绝对坐标无对齐, Table 3 对标)")
        print(f"  PA-MPJPE:   {avg_pampjpe:.1f} mm   (Procrustes)")
        print(f"  pelvisMPJPE:{avg_mpjpe_pelvis:.1f} mm   (骨盆对齐, 诊断: 纯姿态质量)")
    else:
        print(f"  MPJPE:    {avg_mpjpe:.1f} mm   (骨盆对齐 3D)")
        print(f"  PA-MPJPE: {avg_pampjpe:.1f} mm   (Procrustes)")
    print(f"\n  逐关节 {mpjpe_label} (mm):")
    for name, v in per_joint.items():
        bar = '+' * max(0, int((v - avg_mpjpe) / 10)) if v > avg_mpjpe else ''
        print(f"    {name:15s} {v:7.1f} {bar}")
    print(f"\n  逐动作:")
    print(f"    {'Action':12s} {'Cat':6s} {'Count':>6s} {'MPJPE':>8s} {'PA-MPJPE':>10s}")
    for act, r in sorted(action_results.items()):
        print(f"    {act:12s} {r['category']:6s} {r['count']:6d} "
              f"{r['mpjpe_mm']:7.1f} {r['pampjpe_mm']:9.1f}")
    print(f"{'=' * 60}", flush=True)

    # ---- 保存 ----
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    report = {'checkpoint': os.path.abspath(args.ckpt),
              'protocol': config['protocol'],
              'split': config['split_to_use'],
              'target_space': target_space,
              'config': {'path': config_path, 'source': config_source,
                         'fingerprint': actual_fingerprint,
                         'checkpoint_fingerprint': expected_fingerprint},
              'test_identity': test_identity,
              'test_samples': int(N), 'eval_time_s': round(eval_time, 1),
              'mpjpe_mm': round(avg_mpjpe, 2), 'pampjpe_mm': round(avg_pampjpe, 2),
              'mpjpe_pelvis_mm': round(avg_mpjpe_pelvis, 2),
              'test_loss': round(avg_loss, 6),
              'per_joint_mpjpe_mm': per_joint,
              'per_action': action_results}
    out_path = os.path.join(ckpt_dir, 'test_report_detailed.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 详细报告: {out_path}", flush=True)


if __name__ == '__main__':
    main()
