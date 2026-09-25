"""
mmfi_wifi.engine — 训练/评估引擎
==================================
train.py 与 run_all.py 共用的核心循环。

要点:
  - 训练目标由 config.target_space 决定 (损失均为纯 MSE, 无 confidence 加权):
      * absolute      — 绝对相机系 3D 坐标 (米)。**论文 Table 3 复现口径**:
                        官方 mmfi_lib/evaluate.calulate_error 的 MPJPE 无任何对齐,
                        模型必须自己从 CSI 推断人体绝对位置 (场景相关的多径特征),
                        S3 的 MPJPE 爆炸正来自 E04 场景 239-326mm 的位置平移。
      * root_relative — 骨盆相对坐标 (旧口径, 兼容旧 checkpoint; 三划分会趋同,
                        不能复现 Table 3 的 S1<S2<S3 梯度)
  - 评估: absolute → 官方口径 (无对齐 MPJPE + Procrustes 含缩放 PA),
          另记录骨盆对齐 MPJPE 作诊断列 (mpjpe_pelvis_mm);
          root_relative → 骨盆对齐 MPJPE (对预测重新对齐骨盆) + PA
  - 3-way 划分: 从训练集划出 val_fraction 做模型选择 (best epoch / early stopping),
    原始 val 集作为 held-out 测试集, 仅最终评估使用一次 —— 消除 val=test 选择泄漏
  - 抗过拟合: decoder dropout + CSI 数据增强 (高斯噪声 + 幅度缩放)
  - AMP (fp16) + TF32 + Fused AdamW + 梯度裁剪
"""

import os
import csv
import copy
import shutil
import json
import time
import hashlib
import importlib.util
import math
import random
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .experiment_config import write_experiment_config
from .data import make_dataset, make_dataloader, make_manual_test_dataset
from .label_manifest_contract import load_bound_label_manifest, write_bound_label_manifest
from .metrics import pelvis_of
from .pose_metrics import EvaluationOutput, evaluate_pose_triplet
from .checkpoint_selection import (
    CHECKPOINT_FILENAMES,
    BestCheckpointTracker,
    CheckpointRole,
    checkpoint_selection_metadata,
    ROLE_METRIC_NAMES,
    metrics_as_dict,
)
from .legacy_checkpoint import load_metafi_state_dict, strip_orig_mod_prefix
from .model import posenet, weights_init
from .sequence_keys import SequenceKey, partition_train_select, sequence_key_from_item
from .data_manifest import DataManifest, audit_manifest
from .run_identity import (
    NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION,
    RunIdentity,
    assert_new_result_root,
    _assert_loadable_last_state,
    _load_exact_identity,
)


# ============================================================
# torch.compile 兼容工具
# ============================================================
# torch.compile 会把原模型包成 OptimizedModule, 并给 state_dict 键统一加
# ``_orig_mod.`` 前缀。若直接保存/加载会破坏 checkpoint 与 --resume / test.py
# 的兼容性。以下两个工具保证 checkpoint 始终以“未编译”的裸键存取。
def unwrap_model(m):
    """返回被 torch.compile 包裹前的原始模块 (未编译时原样返回)。"""
    return getattr(m, '_orig_mod', m)


def _strip_orig_mod(sd):
    """向后兼容的 torch.compile state-dict 前缀去除入口。"""
    return strip_orig_mod_prefix(sd)


# ============================================================
# CUDA 设置
# ============================================================
def setup_cuda(device, verbose=True):
    if device.type != 'cuda':
        return
    # expandable_segments: 解决 8GB 显存下训练/验证交替导致的内存碎片化 OOM。
    # 典型症状: 空闲 2.87GiB 却无法分配 52MiB (2026-08-02 实测)。
    # 必须在首次 CUDA 分配前设置才生效。
    # 注意: Windows WDDM 不支持 expandable_segments (会触发 UserWarning 但无害),
    # Linux 上则真正生效。保留设置以兼容两种平台。
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable_segments' not in alloc_conf:
        new_val = ('expandable_segments:True,' + alloc_conf) if alloc_conf else 'expandable_segments:True'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = new_val
    # benchmark=False: 在 8GB 显存约束下, benchmark 模式为每种 conv 形状缓存
    # 算法工作空间 (ResNet34 约 1.5-2GB 额外占用), 导致实际训练 OOM。
    # 启发式选择 (benchmark=False) 在现代 cuDNN 上性能差距 <5%,
    # 但节省大量显存, 是 8GB 卡的必要妥协 (2026-08-02 实测确认)。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if verbose:
        print(f"[CUDA] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB) "
              f"| benchmark=OFF TF32=ON", flush=True)


def root_relative(gt):
    """gt: (B, 17, 3) torch -> 骨盆相对坐标 (B, 17, 3)"""
    pelvis = 0.5 * (gt[:, 11, :] + gt[:, 12, :])  # (B, 3)
    return gt - pelvis.unsqueeze(1)


def resolve_target_space(config):
    """训练/评估目标坐标系。
    默认 root_relative: 旧 run 目录的 config.yaml 没有 target_space 键,
    其 checkpoint 是根相对口径训练的, 默认值必须与之一致才能被 test.py 正确评估。
    论文复现请在 config 中显式写 target_space: absolute。
    """
    space = config.get('target_space', 'root_relative')
    if space not in ('absolute', 'root_relative'):
        raise ValueError(f"target_space 必须是 absolute 或 root_relative, 实际 {space!r}")
    return space


def make_target(gt, target_space):
    """gt: (B,17,3) 绝对坐标 -> 训练回归目标。"""
    return gt if target_space == 'absolute' else root_relative(gt)


def augment_csi(csi, noise_std=0.01, scale_range=(0.9, 1.1)):
    """CSI 数据增强: 高斯噪声 + 幅度缩放。仅训练时调用。
    csi: (B, 1, 3, 114, 10) on device。不 clip —— 让模型适应轻微越界值。
    """
    B = csi.shape[0]
    noise = torch.randn_like(csi) * noise_std
    scale = torch.empty(B, 1, 1, 1, 1, device=csi.device,
                        dtype=csi.dtype).uniform_(*scale_range)
    return (csi + noise) * scale


def config_fingerprint(config):
    """训练语义配置的稳定指纹，用于拒绝不兼容的 resume/test。"""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def sequence_key(item):
    """MM-Fi 中一个不可拆分的视频/CSI 序列的身份。"""
    return item['scene'], item['subject'], item['action']


def split_train_select_by_sequence(dataset, val_fraction=0.1, seed=42):
    """按完整 ``(scene, subject, action)`` 序列划分 train/select。

    保留 legacy public API，但将确定性的动作分层配额分配委托给
    :func:`partition_train_select`。返回的 subset 索引顺序与旧实现一致。
    """
    if not hasattr(dataset, 'data_list'):
        raise TypeError("序列级划分要求 dataset 具有 data_list")

    groups: dict[SequenceKey, list[int]] = {}
    for index, item in enumerate(dataset.data_list):
        key = sequence_key_from_item(item)
        groups.setdefault(key, []).append(index)

    train_keys, select_keys = partition_train_select(
        groups,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_indices = [index for key in sorted(train_keys) for index in groups[key]]
    select_indices = [index for key in sorted(select_keys) for index in groups[key]]
    if len(train_indices) + len(select_indices) != len(dataset):
        raise RuntimeError("序列级划分未完整覆盖原训练集")

    split_payload = {
        'unit': 'sequence', 'seed': seed, 'val_fraction': val_fraction,
        'train_keys': [[key.scene, key.subject, key.action] for key in sorted(train_keys)],
        'select_keys': [[key.scene, key.subject, key.action] for key in sorted(select_keys)],
    }
    split_hash = hashlib.sha256(json.dumps(
        split_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]
    metadata = {
        'selection_unit': 'sequence',
        'split_fingerprint': split_hash,
        'train_sequences': len(train_keys),
        'select_sequences': len(select_keys),
        'train_samples': len(train_indices),
        'select_samples': len(select_indices),
    }
    return (torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, select_indices), metadata)

def _atomic_torch_save(state, path):
    """同目录临时文件 + replace，避免中断留下半个 checkpoint。"""
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)





def _atomic_json_write(path, payload):
    """Publish strict JSON only after its complete temporary file is flushed."""
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_npz_write(path, *, predictions, targets, sequence_ids, frame_indices):
    """Atomically persist the absolute-role held-out outputs without pickle data."""
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'wb') as handle:
            np.savez_compressed(
                handle,
                predictions=np.asarray(predictions, dtype=np.float32),
                targets=np.asarray(targets, dtype=np.float32),
                sequence_ids=np.asarray(sequence_ids, dtype=str),
                frame_indices=np.asarray(frame_indices, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _metrics_json(metrics):
    values = metrics_as_dict(metrics)
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('测试指标包含非有限值')
    return values


def _collect_test_identifiers(test_loader, expected_samples):
    """Return one immutable logical sequence ID and frame index per test output row."""
    sequence_ids, frame_indices = [], []
    for batch in test_loader:
        if not isinstance(batch, Mapping):
            raise ValueError('测试 batch 必须是 Mapping，无法导出 sequence identifiers')
        try:
            scenes = batch['scene']
            subjects = batch['subject']
            actions = batch['action']
            indices = batch['idx']
        except KeyError as error:
            raise ValueError('测试 batch 缺少 sequence identifier 字段') from error
        if indices is None:
            raise ValueError('测试 batch 缺少 frame indices')
        try:
            batch_size = len(indices)
        except TypeError as error:
            raise ValueError('测试 batch frame indices 必须是序列') from error
        for values, name in ((scenes, 'scene'), (subjects, 'subject'), (actions, 'action')):
            try:
                if len(values) != batch_size:
                    raise ValueError(f'测试 batch {name} 长度与 idx 不一致')
            except TypeError as error:
                raise ValueError(f'测试 batch {name} 必须是序列') from error
        for scene, subject, action, frame_index in zip(scenes, subjects, actions, indices):
            if not all(isinstance(value, str) and value for value in (scene, subject, action)):
                raise ValueError('测试 batch sequence identifier 无效')
            if not isinstance(frame_index, Integral) or isinstance(frame_index, bool):
                raise ValueError('测试 batch frame index 无效')
            sequence_ids.append(f'{scene}/{subject}/{action}')
            frame_indices.append(int(frame_index))
    if len(sequence_ids) != expected_samples:
        raise ValueError(
            f'测试 sequence identifiers 数量 {len(sequence_ids)} 与预测数量 {expected_samples} 不一致'
        )
    return sequence_ids, frame_indices


def _assert_no_tmp_artifacts(result_dir):
    tmp_files = []
    for root, _dirs, files in os.walk(result_dir):
        for filename in files:
            if filename.endswith('.tmp'):
                tmp_files.append(os.path.join(root, filename))
    if tmp_files:
        raise ValueError(f'完成前发现未提交临时文件: {sorted(tmp_files)!r}')


def _load_role_checkpoint(model, checkpoint_path, role, device, model_factory):
    """Strictly load one selected role checkpoint.

    Finalization must reject every missing/unexpected parameter.  Permissive
    loading can preserve a preceding role's parameter values and make a role
    summary describe a mixed model rather than its own checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'缺少 {os.path.basename(checkpoint_path)}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f'{os.path.basename(checkpoint_path)} 必须是 Mapping')
    if checkpoint.get('checkpoint_role') != role.value:
        raise ValueError(f'{os.path.basename(checkpoint_path)} checkpoint_role 不匹配')
    if checkpoint.get('selected_metric') != ROLE_METRIC_NAMES[role]:
        raise ValueError(f'{os.path.basename(checkpoint_path)} selected_metric 不匹配')
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f'{os.path.basename(checkpoint_path)} model_state_dict 不能为空')

    # This supports legacy flat MetaFi keys through the existing adapter, but
    # unlike ordinary compatibility loading it rejects *all* missing keys,
    # including historical output-affine omissions, for final role evaluation.
    missing, unexpected = load_metafi_state_dict(unwrap_model(model), state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'{os.path.basename(checkpoint_path)} model_state_dict 与当前模型不完整: '
            f'missing={sorted(missing)} unexpected={sorted(unexpected)}'
        )
    return checkpoint


def _task7_staging_dir(result_dir):
    return os.path.join(result_dir, '.finalize-staging')


def _remove_task7_final_artifacts(result_dir):
    """Remove only Task-7 published products, leaving resumable state intact."""
    for filename in (
        'summary_absolute.json', 'summary_pelvis.json', 'summary_pa.json',
        'final_report.json', 'test_outputs_absolute.npz', 'done.txt',
    ):
        artifact = os.path.join(result_dir, filename)
        if os.path.isfile(artifact):
            os.remove(artifact)
    staging_dir = _task7_staging_dir(result_dir)
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)


def _publish_task7_staged_artifacts(result_dir, staging_dir):
    filenames = (
        'summary_absolute.json', 'summary_pelvis.json', 'summary_pa.json',
        'final_report.json', 'test_outputs_absolute.npz', 'done.txt',
    )
    missing = [name for name in filenames if not os.path.isfile(os.path.join(staging_dir, name))]
    if missing:
        raise ValueError(f'finalization staging 缺少产物: {missing!r}')
    for filename in filenames:
        os.replace(os.path.join(staging_dir, filename), os.path.join(result_dir, filename))
    os.rmdir(staging_dir)


def _assert_role_checkpoint_fingerprints(
    checkpoint, *, config_fingerprint_value, execution_fingerprint,
    split_fingerprint, label_manifest_fingerprint,
):
    expected = {
        'config_fingerprint': config_fingerprint_value,
        'execution_fingerprint': execution_fingerprint,
        'split_fingerprint': split_fingerprint,
        'label_manifest_fingerprint': label_manifest_fingerprint,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f'角色 checkpoint {field} 不匹配')


def _validate_metafi_completion(
    *, result_dir, data_manifest, label_manifest, run_identity,
    config_fingerprint_value, execution_fingerprint, split_fingerprint,
    expected_test_samples, artifact_dir=None,
):
    """Fail closed before publishing ``done.txt`` for a new MetaFi SSL run."""
    if not isinstance(expected_test_samples, int) or isinstance(expected_test_samples, bool) or expected_test_samples <= 0:
        raise ValueError('expected_test_samples 必须是正 int')
    audit = audit_manifest(data_manifest, label_manifest.selected_keys)
    if not audit.passed:
        raise ValueError('完成前数据泄露审计失败: ' + '; '.join(audit.violations))
    if run_identity.manifest_fingerprint != data_manifest.fingerprint():
        raise ValueError('完成前 RunIdentity manifest_fingerprint 不匹配')
    if run_identity.label_manifest_fingerprint != label_manifest.fingerprint:
        raise ValueError('完成前 RunIdentity label_manifest_fingerprint 不匹配')
    load_bound_label_manifest(
        Path(result_dir) / 'fewshot_manifest.json',
        expected_identity=run_identity,
        expected_fingerprint=label_manifest.fingerprint,
    )

    identity_path = os.path.join(result_dir, 'run_identity.json')
    saved_identity = _load_exact_identity(Path(identity_path))
    if saved_identity != run_identity:
        raise ValueError('完成前 run_identity.json 不匹配')

    last_path = os.path.join(result_dir, 'last_state.pth')
    _assert_loadable_last_state(Path(last_path), run_identity)
    last_state = torch.load(last_path, map_location='cpu', weights_only=True)
    for field, value in {
        'config_fingerprint': config_fingerprint_value,
        'execution_fingerprint': execution_fingerprint,
        'split_fingerprint': split_fingerprint,
        'label_manifest_fingerprint': label_manifest.fingerprint,
    }.items():
        if last_state.get(field) != value:
            raise ValueError(f'完成前 last_state {field} 不匹配')

    for role, filename in CHECKPOINT_FILENAMES.items():
        checkpoint = torch.load(os.path.join(result_dir, filename), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f'完成前 {filename} 必须是 Mapping')
        if checkpoint.get('checkpoint_role') != role.value:
            raise ValueError(f'完成前 {filename} checkpoint_role 不匹配')
        _assert_role_checkpoint_fingerprints(
            checkpoint,
            config_fingerprint_value=config_fingerprint_value,
            execution_fingerprint=execution_fingerprint,
            split_fingerprint=split_fingerprint,
            label_manifest_fingerprint=label_manifest.fingerprint,
        )

    artifact_dir = result_dir if artifact_dir is None else os.fspath(artifact_dir)
    required_state = ['metrics.csv', 'last_state.pth', *CHECKPOINT_FILENAMES.values()]
    missing = [filename for filename in required_state if not os.path.isfile(os.path.join(result_dir, filename))]
    required_artifacts = [
        'summary_absolute.json', 'summary_pelvis.json', 'summary_pa.json',
        'final_report.json', 'test_outputs_absolute.npz',
    ]
    missing.extend(filename for filename in required_artifacts if not os.path.isfile(os.path.join(artifact_dir, filename)))
    if missing:
        raise FileNotFoundError('完成前缺少必需产物: ' + ', '.join(missing))
    with np.load(os.path.join(artifact_dir, 'test_outputs_absolute.npz'), allow_pickle=False) as outputs:
        expected_keys = {'predictions', 'targets', 'sequence_ids', 'frame_indices'}
        if set(outputs.files) != expected_keys:
            raise ValueError('test_outputs_absolute.npz 字段不匹配')
        if outputs['predictions'].dtype != np.float32 or outputs['targets'].dtype != np.float32:
            raise ValueError('test_outputs_absolute.npz predictions/targets 必须是 float32')
        lengths = [len(outputs[name]) for name in expected_keys]
        if any(length != expected_test_samples for length in lengths):
            raise ValueError('test_outputs_absolute.npz 样本数量不匹配')
    _assert_no_tmp_artifacts(result_dir)
    if artifact_dir != result_dir:
        _assert_no_tmp_artifacts(artifact_dir)

def finalize_metafi_run(
    *, result_dir, model, test_loader, device, use_amp, criterion, target_space,
    model_factory, data_manifest, label_manifest, run_identity,
    config_fingerprint, execution_fingerprint, split_fingerprint,
    expected_test_samples, train_time_s, selection_metric, evaluate_fn=None,
    protocol=None, split=None, amp_dtype=torch.bfloat16, use_channels_last=False,
):
    """Evaluate role checkpoints and publish all Task-7 artifacts transactionally."""
    if not isinstance(run_identity, RunIdentity):
        raise TypeError('run_identity 必须是 RunIdentity')
    if data_manifest is None or label_manifest is None:
        raise ValueError('MetaFi finalization 必须提供 data_manifest 和 label_manifest')
    evaluate_callable = evaluate if evaluate_fn is None else evaluate_fn
    if not callable(evaluate_callable):
        raise TypeError('evaluate_fn 必须可调用')
    result_dir = os.fspath(result_dir)
    assert_new_result_root(Path(result_dir))
    os.makedirs(result_dir, exist_ok=True)
    if run_identity.label_manifest_fingerprint != label_manifest.fingerprint:
        raise ValueError('finalization RunIdentity label_manifest_fingerprint 不匹配')
    load_bound_label_manifest(
        Path(result_dir) / 'fewshot_manifest.json',
        expected_identity=run_identity,
        expected_fingerprint=label_manifest.fingerprint,
    )

    # Remove only stale finalization products.  Keep training checkpoints and
    # last_state so a failed finalization remains resumable.
    _remove_task7_final_artifacts(result_dir)
    staging_dir = _task7_staging_dir(result_dir)
    os.makedirs(staging_dir, exist_ok=False)

    try:
        for role, filename in CHECKPOINT_FILENAMES.items():
            checkpoint_path = os.path.join(result_dir, filename)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'缺少 {filename}')

        role_summaries = {}
        absolute_outputs = None
        for role, filename in CHECKPOINT_FILENAMES.items():
            # A fresh copy prevents any role from inheriting in-memory weights
            # from a preceding role even if callers supplied a shared model.
            role_model = copy.deepcopy(model).to(device)
            checkpoint_path = os.path.join(result_dir, filename)
            checkpoint = _load_role_checkpoint(role_model, checkpoint_path, role, device, model_factory)
            _assert_role_checkpoint_fingerprints(
                checkpoint,
                config_fingerprint_value=config_fingerprint,
                execution_fingerprint=execution_fingerprint,
                split_fingerprint=split_fingerprint,
                label_manifest_fingerprint=label_manifest.fingerprint,
            )
            evaluation = evaluate_callable(
                role_model, test_loader, device, use_amp, criterion,
                target_space=target_space, amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
            )
            if not isinstance(evaluation, EvaluationOutput):
                raise TypeError('evaluate_fn 必须返回 EvaluationOutput')
            test_metrics = _metrics_json(evaluation.metrics)
            test_samples = int(len(evaluation.predictions))
            if len(evaluation.targets) != test_samples or test_samples != expected_test_samples:
                raise ValueError('held-out test 样本数量不匹配')
            summary = {
                'schema_version': 1,
                'checkpoint_role': role.value,
                'checkpoint_file': filename,
                'selected_epoch': checkpoint['selected_epoch'],
                'selected_metric': checkpoint['selected_metric'],
                'selected_value': checkpoint['selected_value'],
                'selection_metrics': checkpoint['metrics'],
                'test_loss': float(evaluation.average_loss),
                'test_metrics': test_metrics,
                'test_samples': test_samples,
                'protocol': protocol if protocol is not None else data_manifest.protocol,
                'split': split if split is not None else data_manifest.split,
                'target_space': target_space,
                'selection_metric': selection_metric,
                'config_fingerprint': config_fingerprint,
                'execution_fingerprint': execution_fingerprint,
                'split_fingerprint': split_fingerprint,
                'manifest_fingerprint': data_manifest.fingerprint(),
                'label_manifest_fingerprint': (label_manifest.fingerprint if label_manifest is not None else None),
                'run_identity_fingerprint': hashlib.sha256(
                    run_identity.canonical_json().encode('utf-8')
                ).hexdigest(),
                'train_time_s': float(train_time_s),
            }
            _atomic_json_write(os.path.join(staging_dir, f'summary_{role.value}.json'), summary)
            role_summaries[role.value] = summary
            if role is CheckpointRole.ABSOLUTE:
                sequence_ids, frame_indices = _collect_test_identifiers(test_loader, test_samples)
                absolute_outputs = (
                    evaluation.predictions, evaluation.targets, sequence_ids, frame_indices,
                )

        if absolute_outputs is None:
            raise RuntimeError('absolute checkpoint finalization 未产生输出')
        _atomic_npz_write(
            os.path.join(staging_dir, 'test_outputs_absolute.npz'),
            predictions=absolute_outputs[0], targets=absolute_outputs[1],
            sequence_ids=absolute_outputs[2], frame_indices=absolute_outputs[3],
        )
        final_report = {
            'schema_version': 1,
            'primary_checkpoint_role': CheckpointRole.ABSOLUTE.value,
            'roles': role_summaries,
        }
        _atomic_json_write(os.path.join(staging_dir, 'final_report.json'), final_report)

        _validate_metafi_completion(
            result_dir=result_dir,
            artifact_dir=staging_dir,
            data_manifest=data_manifest,
            label_manifest=label_manifest,
            run_identity=run_identity,
            config_fingerprint_value=config_fingerprint,
            execution_fingerprint=execution_fingerprint,
            split_fingerprint=split_fingerprint,
            expected_test_samples=expected_test_samples,
        )
        _atomic_json_write(
            os.path.join(staging_dir, 'done.txt'),
            {
                'schema_version': 1,
                'status': 'complete',
                'primary_checkpoint_role': CheckpointRole.ABSOLUTE.value,
                'final_report': 'final_report.json',
            },
        )
        _publish_task7_staged_artifacts(result_dir, staging_dir)
        return final_report
    except Exception:
        _remove_task7_final_artifacts(result_dir)
        raise




class _EpochStrategyScheduler:
    """Deterministic epoch scheduler for identity-bound MetaFi fine-tuning.

    Its stable optimizer group topology keeps optimizer state loadable across
    transfer phase changes.  The selected strategy controls which named groups
    are active; inactive groups receive exactly zero learning rate.
    """

    def __init__(self, optimizer, *, num_epochs, warmup_epochs, lr_min, kind="cosine", milestones=(), gamma=0.5):
        self.optimizer = optimizer
        self.num_epochs = int(num_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.lr_min = float(lr_min)
        self.schedule_kind = str(kind).lower()
        if self.schedule_kind not in {"cosine", "multistep", "constant"}:
            raise ValueError("scheduler must be cosine, multistep, or constant")
        self.milestones = tuple(int(value) for value in milestones)
        if any(value < 0 for value in self.milestones) or tuple(sorted(self.milestones)) != self.milestones:
            raise ValueError("lr_milestones must be sorted non-negative integers")
        self.gamma = float(gamma)
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("lr_gamma must be in (0, 1]")
        self.group_names = tuple(str(group["name"]) for group in optimizer.param_groups)
        if len(self.group_names) != len(set(self.group_names)):
            raise ValueError("MetaFi strategy optimizer parameter groups must have unique names")
        self.last_epoch = -1

    def _factor(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        if self.schedule_kind == "constant":
            return 1.0
        if self.schedule_kind == "multistep":
            return self.gamma ** sum(epoch >= milestone for milestone in self.milestones)
        cosine_epochs = max(self.num_epochs - self.warmup_epochs, 1)
        progress = min(max(epoch - self.warmup_epochs, 0), cosine_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress / cosine_epochs))

    def apply(self, epoch: int, active_groups) -> None:
        active_lrs = {str(group["name"]): float(group["lr"]) for group in active_groups}
        if not set(active_lrs).issubset(set(self.group_names)):
            raise ValueError("strategy returned an unknown optimizer parameter group")
        factor = self._factor(epoch)
        for group in self.optimizer.param_groups:
            base_lr = active_lrs.get(str(group["name"]))
            group["lr"] = 0.0 if base_lr is None else max(self.lr_min, base_lr * factor)
        self.last_epoch = int(epoch)

    def state_dict(self):
        return {
            "kind": "metafi_epoch_strategy",
            "num_epochs": self.num_epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr_min": self.lr_min,
            "schedule_kind": self.schedule_kind,
            "milestones": list(self.milestones),
            "gamma": self.gamma,
            "group_names": list(self.group_names),
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise ValueError("strategy scheduler state must be a mapping")
        legacy = "schedule_kind" not in state
        if legacy and self.schedule_kind != "cosine":
            raise ValueError("legacy strategy scheduler state is only compatible with cosine")
        expected = {
            "kind": "metafi_epoch_strategy",
            "num_epochs": self.num_epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr_min": self.lr_min,
            "group_names": list(self.group_names),
        }
        if not legacy:
            expected.update({
                "schedule_kind": self.schedule_kind,
                "milestones": list(self.milestones),
                "gamma": self.gamma,
            })
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"strategy scheduler state mismatch for {field}")
        last_epoch = state.get("last_epoch")
        if not isinstance(last_epoch, int) or isinstance(last_epoch, bool) or last_epoch < -1:
            raise ValueError("strategy scheduler state has invalid last_epoch")
        self.last_epoch = last_epoch


def build_standard_optimizer(
    parameters,
    config,
    *,
    num_epochs,
    warmup_epochs,
    lr_min,
    fused=False,
):
    """Build the configured optimizer and scheduler for the standard pipeline."""

    name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay, fused=bool(fused)
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optimizer must be adamw or sgd")

    schedule = str(config.get("scheduler", "multistep" if name == "sgd" else "cosine")).lower()
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(num_epochs) - int(warmup_epochs), 1), eta_min=float(lr_min)
        )
    elif schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.get("lr_milestones", [20, 40, 60, 80])),
            gamma=float(config.get("lr_gamma", 0.5)),
        )
    elif schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    else:
        raise ValueError("scheduler must be cosine, multistep, or constant")
    return optimizer, scheduler


def build_training_optimizer(
    parameter_groups,
    config,
    *,
    num_epochs,
    warmup_epochs,
    lr_min,
    fused=False,
):
    """Build the configured optimizer and epoch-aware strategy scheduler."""

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    groups = [dict(group) for group in parameter_groups]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay, fused=bool(fused))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            groups,
            momentum=float(config.get("sgd_momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optimizer must be adamw or sgd")

    scheduler = _EpochStrategyScheduler(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        lr_min=lr_min,
        kind=str(config.get("scheduler", "cosine")),
        milestones=config.get("lr_milestones", ()),
        gamma=float(config.get("lr_gamma", 0.5)),
    )
    return optimizer, scheduler


def _strategy_static_groups(model, strategy):
    """Build one stable all-parameter optimizer partition for a strategy."""
    model.train()
    # Transfer's final phase exposes every encoder partition.  Matched and
    # Sup-DifferentialLR are also complete at epoch eight.
    groups = strategy.apply(unwrap_model(model), epoch=8)
    return [dict(group) for group in groups]


def _strategy_state(strategy, last_applied_epoch):
    if not is_dataclass(strategy):
        raise TypeError("MetaFi fine-tune strategy must be a dataclass")
    if not isinstance(last_applied_epoch, int) or last_applied_epoch < -1:
        raise ValueError("strategy last_applied_epoch is invalid")
    return {
        "name": str(strategy.name),
        "parameters": asdict(strategy),
        "last_applied_epoch": last_applied_epoch,
    }


def _restore_strategy_state(state, strategy, *, next_epoch):
    if not isinstance(state, Mapping):
        raise ValueError("last_state strategy_state must be a mapping")
    expected = _strategy_state(strategy, next_epoch - 1)
    if state != expected:
        raise ValueError("last_state strategy_state is incompatible with this run")


def _apply_new_pipeline_strategy(model, strategy, scheduler, epoch):
    """Apply strategy after ``model.train`` so frozen BN remains in eval mode."""
    model.train()
    active_groups = strategy.apply(unwrap_model(model), epoch=epoch)
    scheduler.apply(epoch, active_groups)
    return active_groups


def _numpy_rng_state_dict():
    bit_generator, values, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "values": values.astype(np.uint32, copy=False).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(state):
    if not isinstance(state, Mapping):
        raise ValueError("last_state numpy_rng_state must be a mapping")
    try:
        bit_generator = state["bit_generator"]
        values = np.asarray(state["values"], dtype=np.uint32)
        position = state["position"]
        has_gauss = state["has_gauss"]
        cached_gaussian = state["cached_gaussian"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("last_state numpy_rng_state is invalid") from error
    if (
        not isinstance(bit_generator, str)
        or not isinstance(position, int)
        or not isinstance(has_gauss, int)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        raise ValueError("last_state numpy_rng_state is invalid")
    np.random.set_state((bit_generator, values, position, has_gauss, float(cached_gaussian)))


def _ensure_fresh_result_dir(result_dir, resume):
    """fresh run 禁止混写；仅 config.yaml 的未启动目录可安全重试。"""
    if resume:
        state_path = os.path.join(result_dir, 'last_state.pth')
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"无法续训，缺少 {state_path}")
        return
    if os.path.isdir(result_dir):
        entries = set(os.listdir(result_dir))
        restartable_files = {
            'config.yaml', 'execution.json', 'metrics.csv',
            'best_model.pth', 'best_model.pth.tmp',
            'best_absolute.pth', 'best_absolute.pth.tmp',
            'best_pelvis.pth', 'best_pelvis.pth.tmp',
            'best_pa.pth', 'best_pa.pth.tmp', 'last_state.pth.tmp',
            'run_identity.json', 'experiment_config.json', 'data_manifest.json',
            'leakage_audit.json', 'label_stats.json', 'pretrain_manifest.json',
        }
        restartable = entries <= restartable_files and 'last_state.pth' not in entries
        if entries and not restartable:
            raise FileExistsError(
                f"结果目录非空，拒绝混写: {result_dir}。请换新目录或使用 --resume。")
    os.makedirs(result_dir, exist_ok=True)


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, use_amp, criterion, max_batches=None,
             target_space='absolute', amp_dtype=torch.bfloat16,
             use_channels_last=False):
    """返回 typed :class:`EvaluationOutput`。

    absolute      : absolute_mpjpe = 官方无对齐 MPJPE (Table 3 口径),
                    pelvis_mpjpe = 骨盆对齐 MPJPE (诊断: 姿态质量与位置误差分离)
    root_relative : absolute_mpjpe = 骨盆对齐 MPJPE (对预测重新对齐),
                    pelvis_mpjpe 与其相同

    ``EvaluationOutput`` 保留 legacy 六值 unpacking，因此现有内部脚本可在
    迁移期间继续写 ``loss, mpjpe, pa, pelvis, preds, gts = evaluate(...)``。
    amp_dtype / use_channels_last 与训练循环保持一致 (默认值保证向后兼容)。
    """
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    all_preds, all_gts = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        csi = batch['input_wifi-csi'].unsqueeze(1).to(device, dtype=torch.float, non_blocking=True)
        # channels_last 仅对 4D 张量有效; 本模型输入为 5D (内部才重塑为 4D),
        # 故此处仅在罕见的 4D 情形生效, 5D 时为安全 no-op (详见训练循环注释)。
        if use_channels_last and csi.dim() == 4:
            csi = csi.to(memory_format=torch.channels_last)
        gt = batch['output'].to(device, dtype=torch.float, non_blocking=True)
        target = make_target(gt, target_space)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(csi)
            loss = criterion(pred, target)

        batch_size = pred.shape[0]
        loss_sum += loss.item() * batch_size
        n_samples += batch_size
        all_preds.append(pred.float().cpu().numpy())
        all_gts.append(target.float().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    metrics = evaluate_pose_triplet(preds, gts, target_space)
    avg_loss = loss_sum / max(n_samples, 1)
    return EvaluationOutput(
        average_loss=avg_loss,
        metrics=metrics,
        predictions=preds,
        targets=gts,
    )


# ============================================================
# 训练曲线
# ============================================================
def plot_curves(csv_path, out_path, title, selection_metric='mpjpe'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib 未安装, 跳过曲线绘制", flush=True)
        return

    epochs, train_loss, val_loss, mpjpe, pampjpe, lr = [], [], [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']) if row['val_loss'] != 'nan' else None)
            mpjpe.append(float(row['mpjpe_mm']) if row['mpjpe_mm'] != 'nan' else None)
            pampjpe.append(float(row['pampjpe_mm']) if row['pampjpe_mm'] != 'nan' else None)
            lr.append(float(row['lr']))

    def valid(vals):
        pairs = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        return zip(*pairs) if pairs else ([], [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Curves — {title}', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', alpha=0.7, label='Train')
    ve, vv = valid(val_loss)
    if ve:
        ax.plot(ve, vv, 'r-', linewidth=2, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Loss (MSE, 3D joints)')

    ax = axes[0, 1]
    me, mv = valid(mpjpe)
    pe, pv = valid(pampjpe)
    if me:
        ax.plot(me, mv, 'g-', linewidth=2, label='MPJPE')
    if pe:
        ax.plot(pe, pv, color='orange', linewidth=2, label='PA-MPJPE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Error (mm)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('MPJPE & PA-MPJPE')

    ax = axes[1, 0]
    selected_epochs, selected_values = ((me, mv) if selection_metric == 'mpjpe' else (pe, pv))
    color = 'green' if selection_metric == 'mpjpe' else 'orange'
    metric_label = 'MPJPE' if selection_metric == 'mpjpe' else 'PA-MPJPE'
    if selected_epochs:
        ax.plot(selected_epochs, selected_values, color=color, linewidth=2)
        best_v = min(selected_values)
        best_e = selected_epochs[selected_values.index(best_v)]
        ax.axhline(y=best_v, color=color, linestyle='--', alpha=0.5,
                   label=f'Best: {best_v:.1f}mm (epoch {best_e})')
        ax.legend()
    ax.set_xlabel('Epoch'); ax.set_ylabel(f'{metric_label} (mm)')
    ax.grid(True, alpha=0.3); ax.set_title(f'{metric_label} (Model Selection)')

    ax = axes[1, 1]
    ax.plot(epochs, lr, 'purple', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3); ax.set_title('Learning Rate')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 单次完整实验 (训练 + 验证 + 测试)
# ============================================================
def preflight_train_one_experiment(
    dataset_root,
    config,
    *,
    num_workers=0,
    data_manifest=None,
    label_manifest=None,
    run_identity=None,
):
    """Validate Task-4 data/config boundaries without creating result files.

    ``train_one_experiment`` repeats this setup when training begins. Keeping
    this gate read-only means a CLI validation failure cannot strand a fresh
    ``run_identity.json`` or block a corrected retry.
    """
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")
    if not isinstance(config, dict):
        raise TypeError("config 必须是 dict")
    for key in ('protocol', 'split_to_use', 'train_loader', 'validation_loader'):
        if key not in config:
            raise KeyError(f"config 缺少 {key}")
    for loader_name in ('train_loader', 'validation_loader'):
        loader = config[loader_name]
        if not isinstance(loader, dict) or not isinstance(loader.get('batch_size'), int):
            raise ValueError(f"config {loader_name}.batch_size 必须是 int")
        if loader['batch_size'] <= 0:
            raise ValueError(f"config {loader_name}.batch_size 必须 > 0")

    train_ds_full, configured_val_ds = make_dataset(
        dataset_root, config,
        preload_packed_csi=config.get('preload_packed_csi', False),
        preload_in_workers=config.get('preload_in_workers', False))
    if data_manifest is None:
        return
    if label_manifest is None:
        raise ValueError("data_manifest 模式必须提供 label_manifest")
    if config['protocol'] != data_manifest.protocol:
        raise ValueError("config protocol 与 data_manifest 不一致")
    if config['split_to_use'] != data_manifest.split:
        raise ValueError("config split_to_use 与 data_manifest 不一致")
    if run_identity is not None:
        if run_identity.protocol != data_manifest.protocol:
            raise ValueError("run_identity protocol 与 data_manifest 不一致")
        if run_identity.split != data_manifest.split:
            raise ValueError("run_identity split 与 data_manifest 不一致")
        if run_identity.data_scope != data_manifest.scope:
            raise ValueError("run_identity data_scope 与 data_manifest 不一致")
        if run_identity.manifest_fingerprint != data_manifest.fingerprint():
            raise ValueError("run_identity manifest_fingerprint 不一致")
    label_keys = frozenset(label_manifest.selected_keys)
    audit = audit_manifest(data_manifest, label_keys)
    if not audit.passed:
        raise ValueError("MetaFi SSL 数据边界审计失败: " + "; ".join(audit.violations))

    def require_keys(dataset, keys, name):
        present = {sequence_key_from_item(item) for item in dataset.data_list}
        missing = frozenset(keys) - present
        if missing:
            raise ValueError(f"{name} manifest 序列不在已配置 Dataset 中: {sorted(missing)!r}")
        if not keys:
            raise ValueError(f"{name} Dataset 为空")

    require_keys(train_ds_full, label_keys, "labeled train")
    require_keys(train_ds_full, data_manifest.select_keys, "select")
    require_keys(configured_val_ds, data_manifest.test_keys, "test")


def train_one_experiment(dataset_root, config, result_dir, device,
                         num_workers=8, use_amp=True, val_every=2,
                         max_train_batches=None, max_val_batches=None,
                         log_prefix="", resume=False,
                         model_factory=None, criterion=None,
                         fewshot_fraction=None, fewshot_seed=42,
                         data_manifest: DataManifest | None = None,
                         label_manifest=None,
                         run_identity: RunIdentity | None = None,
                         experiment_context=None, experiment_artifacts=None):
    """
    在 config 指定的 protocol/split 下完成完整训练, 并用 best ckpt 在
    held-out 集上出最终报告。

    model_factory: 可选, 签名 (dropout_p, target_space) -> nn.Module;
        不给时用默认 posenet (MetaFi 血统)。实验二起的 pose_ssl 模型经此注入。
    criterion: 可选损失模块, 接口 (pred, target) -> scalar;
        不给时用 nn.MSELoss。实验二起用 MSE+骨骼长度损失。
    fewshot_fraction: 可选 (实验三), 从训练序列池按序列级分层抽样该比例;
        select/test 集不受影响; 抽样清单落盘 fewshot_manifest.json。
    data_manifest/label_manifest: 新 MetaFi SSL 管线的已审计序列边界。
        同时提供时，训练/选择/测试集合严格按 manifest 建立，且仅
        label_manifest.selected_keys 可以读取训练标签。
    run_identity: 新结果目录的不可变身份；仅用于交叉验证 manifest/config。

    返回: summary dict
    """
    _ensure_fresh_result_dir(result_dir, resume)
    if val_every <= 0:
        raise ValueError("val_every 必须 > 0")
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")
    if max_train_batches is not None and max_train_batches <= 0:
        raise ValueError("max_train_batches 必须 > 0")
    if max_val_batches is not None and max_val_batches <= 0:
        raise ValueError("max_val_batches 必须 > 0")
    if config.get('num_epochs', 50) <= 0:
        raise ValueError("num_epochs 必须 > 0")

    cfg_fingerprint = config_fingerprint(config)
    config_path = os.path.join(result_dir, 'config.yaml')
    if resume or os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = yaml.safe_load(f)
        if config_fingerprint(saved_config) != cfg_fingerprint:
            raise ValueError("当前配置与 run 目录 config.yaml 不一致")
    else:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    execution_spec = {
        'dataset_root': os.path.abspath(dataset_root),
        'num_workers': num_workers,
        'use_amp': bool(use_amp and device.type == 'cuda'),
        'val_every': val_every,
        'max_train_batches': max_train_batches,
        'max_val_batches': max_val_batches,
        'fewshot_fraction': fewshot_fraction,
        'fewshot_seed': fewshot_seed if fewshot_fraction is not None else None,
        'data_manifest_fingerprint': data_manifest.fingerprint() if data_manifest is not None else None,
        'label_manifest_fingerprint': (
            label_manifest.fingerprint if label_manifest is not None else None
        ),
        'run_identity_fingerprint': (
            hashlib.sha256(run_identity.canonical_json().encode('utf-8')).hexdigest()
            if run_identity is not None else None
        ),
    }
    execution_fingerprint = config_fingerprint(execution_spec)
    execution_path = os.path.join(result_dir, 'execution.json')
    if os.path.isfile(execution_path):
        with open(execution_path, 'r', encoding='utf-8') as f:
            saved_execution = json.load(f)
        if config_fingerprint(saved_execution) != execution_fingerprint:
            raise ValueError("当前执行参数与原 run 不一致")
    else:
        with open(execution_path, 'w', encoding='utf-8') as f:
            json.dump(execution_spec, f, indent=2, ensure_ascii=False)

    if label_manifest is not None:
        if run_identity is None:
            raise ValueError("MetaFi label manifest 必须绑定 RunIdentity")
        if run_identity.label_manifest_fingerprint != label_manifest.fingerprint:
            raise ValueError("RunIdentity label_manifest_fingerprint 与当前 label manifest 不一致")
        label_manifest_path = os.path.join(result_dir, 'fewshot_manifest.json')
        if os.path.isfile(label_manifest_path):
            load_bound_label_manifest(
                label_manifest_path,
                expected_identity=run_identity,
                expected_fingerprint=label_manifest.fingerprint,
            )
        elif resume:
            raise FileNotFoundError("resume 缺少 fewshot_manifest.json")
        else:
            write_bound_label_manifest(label_manifest_path, label_manifest, run_identity)
    if not resume:
        # 首个完整 epoch 前中断时，允许同配置重试；只清理由本引擎生成的
        # 未提交训练产物，不触碰配置和执行身份文件。
        for filename in (
            'metrics.csv', 'best_model.pth', 'best_model.pth.tmp',
            'best_absolute.pth', 'best_absolute.pth.tmp',
            'best_pelvis.pth', 'best_pelvis.pth.tmp',
            'best_pa.pth', 'best_pa.pth.tmp', 'last_state.pth.tmp',
        ):
            path = os.path.join(result_dir, filename)
            if os.path.isfile(path):
                os.remove(path)

    P = lambda msg: print(f"{log_prefix}{msg}", flush=True)

    # ---- 数据 (3-way: train → train + select, 原 val → test) ----
    P(f"[数据] 加载 {config['protocol']} / {config['split_to_use']} ...")
    train_ds_full, configured_val_ds = make_dataset(
        dataset_root, config,
        preload_packed_csi=config.get('preload_packed_csi', False),
        preload_in_workers=config.get('preload_in_workers', False))

    val_fraction = config.get('val_fraction', 0.1)
    split_seed = config.get('init_rand_seed', 42)
    # official_alignment 时直接使用官方 train/validation 两集合，不再切出 select。
    official_alignment = bool(config.get('official_alignment', False))
    if data_manifest is not None:
        if label_manifest is None:
            raise ValueError("data_manifest 模式必须提供 label_manifest")
        if config['protocol'] != data_manifest.protocol:
            raise ValueError("config protocol 与 data_manifest 不一致")
        if config['split_to_use'] != data_manifest.split:
            raise ValueError("config split_to_use 与 data_manifest 不一致")
        if run_identity is not None:
            if run_identity.protocol != data_manifest.protocol:
                raise ValueError("run_identity protocol 与 data_manifest 不一致")
            if run_identity.split != data_manifest.split:
                raise ValueError("run_identity split 与 data_manifest 不一致")
            if run_identity.data_scope != data_manifest.scope:
                raise ValueError("run_identity data_scope 与 data_manifest 不一致")
            if run_identity.manifest_fingerprint != data_manifest.fingerprint():
                raise ValueError("run_identity manifest_fingerprint 不一致")
        label_keys = frozenset(label_manifest.selected_keys)
        audit = audit_manifest(data_manifest, label_keys)
        if not audit.passed:
            raise ValueError("MetaFi SSL 数据边界审计失败: " + "; ".join(audit.violations))

        def subset_for_keys(dataset, keys, name):
            present = {sequence_key_from_item(item) for item in dataset.data_list}
            missing = frozenset(keys) - present
            if missing:
                raise ValueError(f"{name} manifest 序列不在已配置 Dataset 中: {sorted(missing)!r}")
            indices = [
                index for index, item in enumerate(dataset.data_list)
                if sequence_key_from_item(item) in keys
            ]
            if not indices:
                raise ValueError(f"{name} Dataset 为空")
            return torch.utils.data.Subset(dataset, indices)

        train_ds = subset_for_keys(train_ds_full, label_keys, "labeled train")
        select_ds = subset_for_keys(train_ds_full, data_manifest.select_keys, "select")
        test_ds = subset_for_keys(configured_val_ds, data_manifest.test_keys, "test")
        split_meta = {
            'selection_unit': 'audited_manifest_sequence',
            'split_fingerprint': data_manifest.fingerprint()[:16],
            'train_sequences': len(label_keys),
            'select_sequences': len(data_manifest.select_keys),
            'train_samples': len(train_ds),
            'select_samples': len(select_ds),
        }
    elif official_alignment:
        train_ds = train_ds_full
        select_ds = configured_val_ds
        test_ds = configured_val_ds
        train_keys = {sequence_key(item) for item in train_ds.data_list}
        val_keys = {sequence_key(item) for item in configured_val_ds.data_list}
        split_meta = {
            'selection_unit': 'official_train_validation',
            'split_fingerprint': 'official-' + config_fingerprint({
                'protocol': config['protocol'], 'split': config['split_to_use']
            }),
            'train_sequences': len(train_keys),
            'select_sequences': len(val_keys),
            'train_samples': len(train_ds),
            'select_samples': len(select_ds),
        }
    elif config['split_to_use'] == 'manual_split':
        # manual_split 已显式提供四人 selection 集，不再二次切训练集。
        train_ds = train_ds_full
        select_ds = configured_val_ds
        test_ds = make_manual_test_dataset(dataset_root, config)
        train_keys = {sequence_key(item) for item in train_ds.data_list}
        select_keys = {sequence_key(item) for item in select_ds.data_list}
        split_payload = {'unit': 'manual_sequence',
                         'train_keys': [list(k) for k in sorted(train_keys)],
                         'select_keys': [list(k) for k in sorted(select_keys)]}
        split_hash = hashlib.sha256(json.dumps(
            split_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]
        split_meta = {'selection_unit': 'manual_sequence',
                      'split_fingerprint': split_hash,
                      'train_sequences': len(train_keys),
                      'select_sequences': len(select_keys),
                      'train_samples': len(train_ds),
                      'select_samples': len(select_ds)}
    else:
        test_ds = configured_val_ds
        train_ds, select_ds, split_meta = split_train_select_by_sequence(
            train_ds_full, val_fraction=val_fraction, seed=split_seed)
    n_train = len(train_ds)
    n_select = len(select_ds)

    # ---- 实验三: 少样本序列级抽样 (select/test 不受影响) ----
    if fewshot_fraction is not None:
        if data_manifest is not None:
            raise ValueError("audited manifest 模式禁止 legacy fewshot_fraction 抽样")
        from pose_ssl.fewshot import sample_train_sequences
        train_ds, fewshot_meta = sample_train_sequences(
            train_ds, fewshot_fraction, seed=fewshot_seed)
        n_train = len(train_ds)
        fewshot_path = os.path.join(result_dir, 'fewshot_manifest.json')
        if not resume or not os.path.isfile(fewshot_path):
            with open(fewshot_path, 'w', encoding='utf-8') as f:
                json.dump(fewshot_meta, f, indent=2, ensure_ascii=False)
        P(f"[少样本] fraction={fewshot_fraction} seed={fewshot_seed}: "
          f"{fewshot_meta['sampled_sequences']}/{fewshot_meta['pool_sequences']} "
          f"序列 ({n_train:,} 帧)")

    bs = config['train_loader']['batch_size']
    vbs = config['validation_loader']['batch_size']
    prefetch_factor = config.get('prefetch_factor', 4)
    train_rng = torch.Generator().manual_seed(split_seed)
    eval_rng = torch.Generator().manual_seed(split_seed + 1)
    train_loader = make_dataloader(train_ds, True, train_rng, bs, num_workers,
                                   prefetch_factor=prefetch_factor)
    select_loader = make_dataloader(select_ds, False, eval_rng, vbs, min(num_workers, 2),
                                    prefetch_factor=prefetch_factor)
    test_loader = make_dataloader(test_ds, False, eval_rng, vbs, min(num_workers, 2),
                                  prefetch_factor=prefetch_factor)
    P(f"[数据] Train: {n_train:,} / {split_meta['train_sequences']} seq "
      f"({len(train_loader)} batches) | Select: {n_select:,} / "
      f"{split_meta['select_sequences']} seq ({len(select_loader)} batches) | "
      f"Test: {len(test_ds):,} ({len(test_loader)} batches) | "
      f"split={split_meta['split_fingerprint']}")

    # ---- 模型 ----
    # 每个 split 独立从相同种子开始，不受 run_all 中前序 split 的 RNG 消耗影响。
    torch.manual_seed(split_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(split_seed)
    dropout_p = config.get('dropout_p', 0.0)
    target_space = resolve_target_space(config)
    # 计算路径优化 (均配置驱动 + 默认值 + 可回滚; 不改变模型数学结构)
    use_channels_last = config.get('use_channels_last', True)
    use_compile = config.get('use_compile', True)
    amp_dtype_str = config.get('amp_dtype', 'bf16')
    amp_dtype = torch.float16 if amp_dtype_str == 'fp16' else torch.bfloat16
    if model_factory is None:
        model = posenet(
            dropout_p=dropout_p,
            target_space=target_space,
            pretrained_backbone=bool(config.get('pretrained_backbone', False)),
        )
        # benchmark 训练脚本没有调用 weights_init；保留 torchvision/官方默认初始化。
        if not official_alignment:
            model.apply(weights_init)
    else:
        # 外部工厂 (pose_ssl 等): 自行负责初始化, 不叠加 weights_init
        model = model_factory(dropout_p=dropout_p, target_space=target_space)
    # channels_last: nn.Module.to(memory_format=...) 仅对 4D/5D 参数生效,
    # 对 conv 权重切换内存布局, 其余参数自动跳过 (安全, 不改数值)。
    if use_channels_last and device.type == 'cuda':
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    # torch.compile: 守卫式启用, 失败自动回退到 eager。必须在构建 optimizer
    # 之前, 使 optimizer 拿到与编译后共享、顺序不变的参数。
    # 注意: inductor 后端是惰性编译 —— torch.compile(...) 调用本身会成功,
    # 真正的编译发生在首次 forward, 且依赖 Triton。Windows 无 Triton 时会在
    # 首次 forward 抛 torch._inductor.exc.TritonMissing (在本 try 作用域之外)。
    # 故此处先预检测 Triton, 不可用则直接跳过 compile (channels_last 仍生效);
    # 首次 forward 另有兜底 (见 GPU 预热 / resume 试探)。
    if use_compile and device.type == 'cuda':
        if importlib.util.find_spec('triton') is None:
            P("[编译] 未检测到 Triton，跳过 torch.compile，使用 eager")
        else:
            try:
                model = torch.compile(model, backend='inductor',
                                      mode='reduce-overhead', dynamic=True)
                P("[编译] torch.compile 已启用 (inductor)")
            except Exception as e:
                P(f"[警告] torch.compile 失败，回退: {e}")
    P(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()):,} | "
      f"dropout={dropout_p} | out_affine={target_space}")

    # ---- 优化器 / 调度器 ----
    lr = config.get('learning_rate', 0.0003)
    wd = config.get('weight_decay', 0.01)
    n_epochs = config.get('num_epochs', 50)
    warmup = config.get('lr_warmup_epochs', 5)
    lr_min = config.get('lr_min', 1e-6)
    patience = config.get('early_stopping_patience', 20)
    new_pipeline = run_identity is not None
    fine_tune_strategy = None
    strategy_scheduler = None
    if new_pipeline:
        from pose_ssl.metafi.fine_tune_strategy import select_fine_tune_strategy

        fine_tune_strategy = select_fine_tune_strategy(run_identity.fine_tune_strategy, config)
    # 按 epoch 早停开关 (默认关以保后向兼容): True 时 patience 语义为 epoch 数,
    # False 时保持原“连续无改善验证次数”语义。
    use_epoch_patience = config.get('use_epoch_patience', False)
    selection_metric = config.get('selection_metric', 'mpjpe').lower()
    if selection_metric not in ('mpjpe', 'pampjpe'):
        raise ValueError("selection_metric 必须是 mpjpe 或 pampjpe")
    fused = device.type == 'cuda'
    optimizer_name = str(config.get('optimizer', 'adamw')).lower()
    if new_pipeline:
        # SSL 微调沿用已有分组策略；其 encoder 已换为官方 MetaFi++ 骨架。
        static_groups = _strategy_static_groups(model, fine_tune_strategy)
        optimizer, strategy_scheduler = build_training_optimizer(
            static_groups,
            config,
            num_epochs=n_epochs,
            warmup_epochs=warmup,
            lr_min=lr_min,
            fused=fused,
        )
        scheduler = strategy_scheduler
    else:
        optimizer, scheduler = build_standard_optimizer(
            model.parameters(),
            config,
            num_epochs=n_epochs,
            warmup_epochs=warmup,
            lr_min=lr_min,
            fused=fused,
        )
    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if criterion is None:
        criterion = nn.MSELoss()
    augment = config.get('augment', False)
    aug_noise = config.get('augment_noise_std', 0.01)
    aug_scale = config.get('augment_scale_range', [0.9, 1.1])
    mpjpe_label = 'officialMPJPE' if target_space == 'absolute' else 'MPJPE'
    schedule_name = str(config.get('scheduler', 'multistep' if optimizer_name == 'sgd' else 'cosine')).lower()
    P(f"[训练] {optimizer_name.upper()} lr={lr} wd={wd} | {schedule_name} | "
      f"official_alignment={official_alignment} | AMP={use_amp} | patience={patience} | "
      f"target={target_space} select={selection_metric.upper()} val_every={val_every} | "
      f"augment={augment} noise={aug_noise} scale={aug_scale}")

    # ---- GPU 预热 (含反向, 提前暴露 cuDNN 问题；不污染 BatchNorm 统计) ----
    # resume 时跳过预热: 续训已有可用权重, 无需重复 smoke。
    write_experiment_config(
        result_dir, stage=("finetune" if run_identity is not None and run_identity.method != "sup" else "supervised"),
        config=config,
        execution={**execution_spec, "device": str(device), "epochs": n_epochs,
                   "batch_size": bs, "amp_dtype": str(amp_dtype),
                   "optimizer": optimizer.state_dict()["param_groups"],
                   "scheduler": scheduler.state_dict() if scheduler is not None else None,
                   "strategy": asdict(fine_tune_strategy) if is_dataclass(fine_tune_strategy) else None},
        context=experiment_context, artifacts=experiment_artifacts,
    )

    if device.type == 'cuda' and not resume:
        P("[预热] 前向+反向 smoke ...")
        was_training = model.training
        model.eval()
        dummy = torch.randn(bs, 1, 3, 114, 10, device=device)
        dummy_gt = torch.randn(bs, 17, 3, device=device)
        try:
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                pred_dummy = model(dummy)
                loss_dummy = criterion(pred_dummy, dummy_gt)
        except Exception as e:
            # inductor 惰性编译在首次 forward 才触发 (如 Windows 无 Triton),
            # 该异常在 compile 调用的 try 作用域之外 —— 此处兜底回退 eager,
            # 并用 eager 重跑一次 dummy 前向确认可用, 保证进程不崩。
            P(f"[警告] 编译在首次前向失败，回退 eager: {e}")
            model = unwrap_model(model)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                pred_dummy = model(dummy)
                loss_dummy = criterion(pred_dummy, dummy_gt)
        scaler.scale(loss_dummy).backward()
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.cuda.synchronize()
        del dummy, dummy_gt, pred_dummy, loss_dummy
        P("[预热] 完成")
    elif device.type == 'cuda' and resume:
        # resume 跳过了预热, 但编译仍是惰性的: 若不处理, 编译异常会在
        # 训练循环首个 batch 才抛出而崩溃。故进入训练循环前做一次轻量
        # 前向试探, 复用同样的“失败即回退 eager”保护, 覆盖 resume 路径。
        # eval 模式且不反向, 不污染 BatchNorm 统计。
        was_training = model.training
        model.eval()
        probe = torch.randn(bs, 1, 3, 114, 10, device=device)
        try:
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                model(probe)
        except Exception as e:
            P(f"[警告] 编译在首次前向失败，回退 eager: {e}")
            model = unwrap_model(model)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                model(probe)
        model.train(was_training)
        torch.cuda.synchronize()
        del probe

    # ---- 日志 / 续训状态 ----
    csv_path = os.path.join(result_dir, 'metrics.csv')
    csv_header = ['epoch', 'train_loss', 'val_loss', 'mpjpe_mm', 'pampjpe_mm',
                  'mpjpe_pelvis_mm', 'lr', 'time_s']
    # Every validation updates the three metric-specific checkpoint roles.
    # ``best_model.pth`` remains a compatibility alias for the role selected by
    # the legacy config.  The default ``mpjpe`` selection therefore aliases
    # ``best_absolute.pth``.  Absolute-based early stopping in the matched
    # pipeline keeps training long enough for the pelvis/PA files to update.
    best_tracker = BestCheckpointTracker()
    selection_role = (
        CheckpointRole.ABSOLUTE
        if selection_metric == 'mpjpe'
        else CheckpointRole.PA
    )
    best_value = float('inf')
    best_pampjpe = float('inf')
    best_mpjpe = float('inf')
    best_epoch = -1
    patience_counter = 0
    start_epoch = 0
    elapsed_before = 0.0

    if resume:
        last_path = os.path.join(result_dir, 'last_state.pth')
        state = torch.load(last_path, map_location=device, weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("last_state 必须是 Mapping")
        if state.get('config_fingerprint') != cfg_fingerprint:
            raise ValueError("last_state 的配置指纹不匹配")
        if state.get('execution_fingerprint') != execution_fingerprint:
            raise ValueError("last_state 的执行参数指纹不匹配")
        if state.get('split_fingerprint') != split_meta['split_fingerprint']:
            raise ValueError("last_state 的序列划分指纹不匹配")
        if new_pipeline and label_manifest is not None and state.get('label_manifest_fingerprint') != label_manifest.fingerprint:
            raise ValueError("last_state 的 label manifest 指纹不匹配")
        if new_pipeline:
            if state.get('schema_version') != NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION:
                raise ValueError(
                    "new MetaFi pipeline last_state schema_version 不兼容"
                )
            try:
                saved_identity = RunIdentity.from_dict(state['run_identity'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("last_state 缺少有效 run_identity") from error
            if saved_identity != run_identity:
                raise ValueError("last_state RunIdentity 不匹配")
            if state.get('run_identity_json') != run_identity.canonical_json():
                raise ValueError("last_state run_identity_json 不规范或不匹配")
            _restore_strategy_state(
                state.get('strategy_state'),
                fine_tune_strategy,
                next_epoch=int(state['next_epoch']),
            )

        if model_factory is None:
            load_metafi_state_dict(
                unwrap_model(model), state['model_state_dict'], strict=True)
        else:
            unwrap_model(model).load_state_dict(
                _strip_orig_mod(state['model_state_dict']), strict=False)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = int(state['next_epoch'])
        tracker_state = state.get('best_records') if new_pipeline else state.get('best_checkpoint_tracker')
        try:
            best_tracker.load_state_dict(tracker_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "last_state 缺少完整 best checkpoint tracker，拒绝不安全续训"
            ) from error
        selected_record = best_tracker.best[selection_role]
        best_value = selected_record.value
        best_mpjpe = selected_record.metrics.absolute_mpjpe_mm
        best_pampjpe = selected_record.metrics.pa_mpjpe_mm
        best_epoch = selected_record.epoch
        patience_counter = int(state['patience_counter'])
        # Legacy scalar fields remain in last_state for older readers, but a
        # resumed run trusts the complete role tracker.  Reject mismatches so
        # a corrupt checkpoint cannot silently reset primary selection.
        if (
            float(state['best_selection_value']) != best_value
            or int(state['best_epoch']) != best_epoch
            or float(state['best_mpjpe_mm']) != best_mpjpe
            or float(state['best_pampjpe_mm']) != best_pampjpe
        ):
            raise ValueError("last_state 的主选模标量与角色 tracker 不一致")
        if use_epoch_patience:
            patience_counter = max(0, (start_epoch - 1) - best_epoch)
        elapsed_before = float(state.get('elapsed_train_time_s', 0.0))
        train_rng.set_state(state['train_generator_state'].cpu())
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if new_pipeline:
            try:
                random.setstate(state['python_rng_state'])
                _restore_numpy_rng_state(state['numpy_rng_state'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("last_state 缺少有效 Python/NumPy RNG 状态") from error
        if device.type == 'cuda' and state.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng_states']])
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("resume 缺少 metrics.csv")
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        committed_rows = [r for r in rows if int(r['epoch']) < start_epoch]
        if len(committed_rows) != len(rows):
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                writer.writerows(committed_rows)
            P(f"[续训] 已移除 {len(rows) - len(committed_rows)} 行未提交 CSV 记录")
        logged_epochs = [int(r['epoch']) for r in committed_rows]
        if logged_epochs and logged_epochs[-1] != start_epoch - 1:
            raise ValueError("metrics.csv 缺少 last_state 已提交 epoch 的记录")
        P(f"[续训] 从 epoch {start_epoch} 恢复 | best E{best_epoch} "
          f"{selection_metric.upper()}={best_value:.1f}mm")
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=csv_header).writeheader()

    def save_best(role, record):
        """Persist one independently selected best checkpoint role."""
        state = {
            'format_version': 3,
            'epoch': record.epoch,
            'model_state_dict': unwrap_model(model).state_dict(),
            **checkpoint_selection_metadata(role, record),
            # Retained for legacy readers of ``best_model.pth``.
            'selection_metric': selection_metric,
            'config_fingerprint': cfg_fingerprint,
            'execution_fingerprint': execution_fingerprint,
            'split_fingerprint': split_meta['split_fingerprint'],
            'label_manifest_fingerprint': (label_manifest.fingerprint if label_manifest is not None else None),
        }
        _atomic_torch_save(
            state,
            os.path.join(result_dir, CHECKPOINT_FILENAMES[role]),
        )
        # Preserve the legacy filename/API.  In default runs this is a second
        # atomic serialization of the same Absolute-selected model state.
        if role is selection_role:
            _atomic_torch_save(state, os.path.join(result_dir, 'best_model.pth'))

    terminal_resume = start_epoch >= n_epochs or (resume and patience_counter >= patience)
    if terminal_resume:
        reason = (f"已到 epoch {start_epoch - 1}" if start_epoch >= n_epochs
                  else f"已满足早停 patience={patience}")
        P(f"[续训] last_state {reason}，无需继续训练，转入最终测试")

    # ==================== 训练循环 ====================
    t_start = time.time()
    epoch_range = range(0) if terminal_resume else range(start_epoch, n_epochs)
    for epoch in epoch_range:
        epoch_start = time.time()

        if new_pipeline:
            active_groups = _apply_new_pipeline_strategy(
                model, fine_tune_strategy, strategy_scheduler, epoch
            )
        else:
            if epoch < warmup:
                warmup_lr = lr * (epoch + 1) / warmup
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
            model.train()
        lr_used = max(float(pg['lr']) for pg in optimizer.param_groups)

        # ---- Train ----
        train_loss_sum = 0.0
        n_train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            csi = batch['input_wifi-csi'].unsqueeze(1).to(device, dtype=torch.float, non_blocking=True)
            # channels_last 仅对 4D 张量有效。本模型输入为 5D (B,1,3,114,10),
            # 内部切片/转置/flatten 后才成 4D 再进入 conv; 真正的布局优化由
            # model.to(memory_format=channels_last) 在 conv 权重层面完成。此处对 5D
            # 输入为安全 no-op (避免 5D 用 channels_last 报错), 仅保留接口一致性。
            if use_channels_last and csi.dim() == 4:
                csi = csi.to(memory_format=torch.channels_last)
            gt = batch['output'].to(device, dtype=torch.float, non_blocking=True)
            target = make_target(gt, target_space)

            if augment:
                csi = augment_csi(csi, noise_std=aug_noise, scale_range=tuple(aug_scale))

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                pred = model(csi)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item()
            n_train_batches += 1
            if bi % 200 == 0:
                P(f"  E{epoch:3d} B{bi:5d}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(n_train_batches, 1)

        # ---- Validation (model-selection set) ----
        do_val = (epoch % val_every == 0) or (epoch == n_epochs - 1)
        if do_val:
            evaluation = evaluate(
                model, select_loader, device, use_amp, criterion, max_val_batches,
                target_space=target_space, amp_dtype=amp_dtype,
                use_channels_last=use_channels_last)
            val_loss = evaluation.average_loss
            mpjpe = evaluation.metrics.absolute_mpjpe_mm
            pampjpe = evaluation.metrics.pa_mpjpe_mm
            mpjpe_pelvis = evaluation.metrics.pelvis_mpjpe_mm
        else:
            val_loss, mpjpe, pampjpe, mpjpe_pelvis = (float('nan'),) * 4

        epoch_time = time.time() - epoch_start
        metrics = {'epoch': epoch,
                   'train_loss': round(avg_train_loss, 6),
                   'val_loss': round(val_loss, 6) if do_val else 'nan',
                   'mpjpe_mm': round(mpjpe, 2) if do_val else 'nan',
                   'pampjpe_mm': round(pampjpe, 2) if do_val else 'nan',
                   'mpjpe_pelvis_mm': round(mpjpe_pelvis, 2) if do_val else 'nan',
                   'lr': round(lr_used, 8),
                   'time_s': round(epoch_time, 1)}
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=csv_header).writerow(metrics)

        # ---- Best / early stop ----
        is_best = False
        if do_val:
            # Save each role independently.  The records continue to update
            # until the legacy primary selection/early-stop policy ends this
            # run; default ``mpjpe`` uses the Absolute role.
            updated_roles = best_tracker.update(
                epoch,
                evaluation.metrics,
                save_best,
            )
            if selection_role in updated_roles:
                selected_record = best_tracker.best[selection_role]
                best_value = selected_record.value
                best_pampjpe = selected_record.metrics.pa_mpjpe_mm
                best_mpjpe = selected_record.metrics.absolute_mpjpe_mm
                best_epoch = selected_record.epoch
                patience_counter = 0
                is_best = True
            else:
                # use_epoch_patience: patience 以 epoch 计; 否则以验证次数计 (原语义)。
                if use_epoch_patience:
                    patience_counter = epoch - best_epoch
                else:
                    patience_counter += 1

        if not new_pipeline and (optimizer_name == 'sgd' or epoch >= warmup):
            scheduler.step()

        elapsed_total = elapsed_before + (time.time() - t_start)
        last_state = {
            'format_version': 3,
            'next_epoch': epoch + 1,
            'model_state_dict': unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_selection_value': best_value,
            'best_mpjpe_mm': best_mpjpe,
            'best_pampjpe_mm': best_pampjpe,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'best_checkpoint_tracker': best_tracker.state_dict(),
            'elapsed_train_time_s': elapsed_total,
            'selection_metric': selection_metric,
            'config_fingerprint': cfg_fingerprint,
            'execution_fingerprint': execution_fingerprint,
            'split_fingerprint': split_meta['split_fingerprint'],
            'label_manifest_fingerprint': (label_manifest.fingerprint if label_manifest is not None else None),
            'train_generator_state': train_rng.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
        }
        if new_pipeline:
            last_state.update({
                'schema_version': NEW_PIPELINE_LAST_STATE_SCHEMA_VERSION,
                'run_identity': run_identity.to_dict(),
                'run_identity_json': run_identity.canonical_json(),
                'strategy_state': _strategy_state(fine_tune_strategy, epoch),
                'best_records': best_tracker.state_dict(),
                'python_rng_state': random.getstate(),
                'numpy_rng_state': _numpy_rng_state_dict(),
            })
        _atomic_torch_save(last_state, os.path.join(result_dir, 'last_state.pth'))

        if do_val:
            P(f"  >>> E{epoch:3d} | loss {avg_train_loss:.4f}/{val_loss:.4f} | "
              f"[select] {mpjpe_label} {mpjpe:.1f}mm PA-MPJPE {pampjpe:.1f}mm "
              f"pelvisMPJPE {mpjpe_pelvis:.1f}mm | "
              f"LR {lr_used:.2e} | {epoch_time:.0f}s{' *BEST*' if is_best else ''}")
        else:
            P(f"  >>> E{epoch:3d} | loss {avg_train_loss:.4f}/skip | "
              f"LR {lr_used:.2e} | {epoch_time:.0f}s")

        if patience_counter >= patience:
            P(f"[早停] epoch {epoch}: {selection_metric.upper()} 连续 "
              f"{patience} 次验证无改善")
            break
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    train_time = elapsed_before + (time.time() - t_start)
    P(f"[训练完成] {train_time / 60:.0f}min | best epoch {best_epoch}: "
      f"MPJPE {best_mpjpe:.1f}mm PA-MPJPE {best_pampjpe:.1f}mm")

    # ==================== 测试 / 完成标记 ====================
    if new_pipeline and data_manifest is not None and label_manifest is not None:
        P('[测试] 逐一加载 Absolute / Pelvis / PA checkpoint 评估 held-out 测试集 ...')
        final_report = finalize_metafi_run(
            result_dir=result_dir,
            model=model,
            test_loader=test_loader,
            device=device,
            use_amp=use_amp,
            criterion=criterion,
            target_space=target_space,
            model_factory=model_factory,
            data_manifest=data_manifest,
            label_manifest=label_manifest,
            run_identity=run_identity,
            config_fingerprint=cfg_fingerprint,
            execution_fingerprint=execution_fingerprint,
            split_fingerprint=split_meta['split_fingerprint'],
            expected_test_samples=len(test_ds),
            train_time_s=train_time,
            selection_metric=selection_metric,
            protocol=config['protocol'],
            split=config['split_to_use'],
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
        )
        primary_summary = dict(final_report['roles'][CheckpointRole.ABSOLUTE.value])
        primary_metrics = primary_summary['test_metrics']
        # Preserve the train_one_experiment return contract for existing callers;
        # all three role-specific artifacts remain the authoritative files.
        primary_summary.update({
            'best_epoch': int(best_epoch),
            'best_selection_value': float(best_value),
            'best_mpjpe_mm': float(best_mpjpe),
            'best_pampjpe_mm': float(best_pampjpe),
            'train_samples': n_train,
            'select_samples': n_select,
            'train_sequences': split_meta['train_sequences'],
            'select_sequences': split_meta['select_sequences'],
            'test_mpjpe_mm': float(primary_metrics['absolute_mpjpe_mm']),
            'test_pampjpe_mm': float(primary_metrics['pa_mpjpe_mm']),
            'test_mpjpe_pelvis_mm': float(primary_metrics['pelvis_mpjpe_mm']),
            'result_dir': result_dir,
        })
        P(f'[输出] {result_dir}/')
        P(
            '       metrics.csv | best_absolute.pth | best_pelvis.pth | best_pa.pth | '
            'summary_absolute.json | summary_pelvis.json | summary_pa.json | '
            'final_report.json | test_outputs_absolute.npz | done.txt'
        )
        summary = primary_summary
    else:
        P('[测试] 加载 best_model.pth 评估 held-out 测试集 ...')
        ckpt = torch.load(os.path.join(result_dir, 'best_model.pth'),
                          map_location=device, weights_only=True)
        if model_factory is None:
            load_metafi_state_dict(
                unwrap_model(model), ckpt['model_state_dict'], strict=True)
        else:
            unwrap_model(model).load_state_dict(
                _strip_orig_mod(ckpt['model_state_dict']), strict=False)
        test_evaluation = evaluate(
            model, test_loader, device, use_amp, criterion, target_space=target_space,
            amp_dtype=amp_dtype, use_channels_last=use_channels_last)
        test_loss = test_evaluation.average_loss
        test_mpjpe = test_evaluation.metrics.absolute_mpjpe_mm
        test_pampjpe = test_evaluation.metrics.pa_mpjpe_mm
        test_mpjpe_pelvis = test_evaluation.metrics.pelvis_mpjpe_mm
        tp = test_evaluation.predictions

        P(f'[测试] loss {test_loss:.4f} | {mpjpe_label} {test_mpjpe:.1f}mm | '
          f'PA-MPJPE {test_pampjpe:.1f}mm | pelvisMPJPE {test_mpjpe_pelvis:.1f}mm')

        summary = {'protocol': config['protocol'], 'split': config['split_to_use'],
                   'config_fingerprint': cfg_fingerprint,
                   'target_space': target_space,
                   'selection_metric': selection_metric,
                   'selection_unit': split_meta['selection_unit'],
                   'split_fingerprint': split_meta['split_fingerprint'],
             'label_manifest_fingerprint': (label_manifest.fingerprint if label_manifest is not None else None),
                   'best_epoch': best_epoch,
                   'best_selection_value': best_value,
                   'train_time_s': round(train_time, 1),
                   'best_mpjpe_mm': best_mpjpe, 'best_pampjpe_mm': best_pampjpe,
                   'train_samples': n_train, 'select_samples': n_select,
                   'train_sequences': split_meta['train_sequences'],
                   'select_sequences': split_meta['select_sequences'],
                   'test_loss': round(test_loss, 6),
                   'test_mpjpe_mm': round(test_mpjpe, 2),
                   'test_pampjpe_mm': round(test_pampjpe, 2),
                   'test_mpjpe_pelvis_mm': round(test_mpjpe_pelvis, 2),
                   'test_samples': int(len(tp)),
                   'result_dir': result_dir}
        with open(os.path.join(result_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(result_dir, 'test_report.json'), 'w', encoding='utf-8') as f:
            json.dump({'target_space': target_space,
                       'test_loss': round(test_loss, 6),
                       'test_mpjpe_mm': round(test_mpjpe, 2),
                       'test_pampjpe_mm': round(test_pampjpe, 2),
                       'test_mpjpe_pelvis_mm': round(test_mpjpe_pelvis, 2),
                       'test_samples': int(len(tp))}, f, indent=2, ensure_ascii=False)

        plot_curves(csv_path, os.path.join(result_dir, 'curves.png'),
                    f"{config['protocol']}/{config['split_to_use']}", selection_metric)
        P(f'[输出] {result_dir}/')
        P(
            '       curves.png | metrics.csv | best_absolute.pth | '
            'best_pelvis.pth | best_pa.pth | best_model.pth (legacy) | '
            'summary.json | test_report.json'
        )


    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return summary
