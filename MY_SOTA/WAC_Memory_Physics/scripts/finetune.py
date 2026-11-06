from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from MY_SOTA.WAC_Memory_Physics.src.model import IMPLEMENTATION_VERSION, PhysAlignPose, PhysicalEncoder
from MY_SOTA.WAC_Memory_Physics.scripts.common import *


def mpjpe(pred, target):
    return (pred.float() - target.float()).norm(dim=-1).mean() * 1000


def pa(pred, target):
    pred, target = pred.float(), target.float()
    pm, tm = pred.mean(-2, keepdim=True), target.mean(-2, keepdim=True)
    p, q = pred - pm, target - tm
    u, s, vh = torch.linalg.svd(p.transpose(-2, -1) @ q)
    d = torch.det(u @ vh).sign().reshape(-1, 1, 1)
    c = torch.eye(3, device=pred.device).expand(len(pred), -1, -1).clone()
    c[:, -1, -1] = d[:, 0, 0]
    rotation = u @ c @ vh
    trace = (s * torch.cat([torch.ones_like(s[..., :2]), d.squeeze(-1)], -1)).sum(-1)[:, None, None]
    scale = trace / p.square().sum((-2, -1), keepdim=True).clamp_min(1e-8)
    return (scale * (p @ rotation) + tm - target).norm(dim=-1).mean() * 1000


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    sums = {}
    frames = 0
    for i, (x, raw, target, _) in enumerate(loader, 1):
        output = model(x.to(device), raw.to(device))
        target = target.to(device)
        if not torch.isfinite(output.pose).all():
            raise RuntimeError('nonfinite evaluation predictions')
        n = len(target)
        metrics = dict(absolute_mpjpe_mm=mpjpe(output.pose, target), pa_mpjpe_mm=pa(output.pose, target),
                       root_mae_mm=(output.root - target[:, 0]).norm(dim=-1).mean() * 1000,
                       anchor_error_mm=(output.anchor - target[:, 0]).norm(dim=-1).mean() * 1000,
                       alignment_mm=output.align.norm(dim=-1).mean() * 1000,
                       memory_confidence=output.weight[:, 0].mean())
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.) + float(value) * n
        frames += n
        if max_batches and i >= max_batches:
            break
    if not frames:
        raise ValueError('empty evaluation set')
    if not max_batches and frames != len(loader.dataset):
        raise RuntimeError('incomplete evaluation')
    return {**{k: v / frames for k, v in sums.items()}, 'frames': frames}


def tensor_fingerprint(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    e, mc = cfg['experiment'], cfg['model']
    run_type = validate_run_paths(cfg)
    seed_all(int(e['seed']))
    manifest, source, label, select, test = load_protocol(cfg)
    chosen = pretrain_selection(cfg, source, label, select, test)
    ids = {seq_string(k): i for i, k in enumerate(sorted(source | select | test))}
    root = resolve_path(e['dataset_root'])
    pre_path = resolve_path(cfg['artifacts']['pretrain_checkpoint'])
    pre_sha = file_fingerprint(pre_path)
    state = torch.load(pre_path, map_location='cpu', weights_only=False)
    premeta = json.loads((pre_path.parent / 'pretrain_manifest.json').read_text(encoding='utf-8'))
    pre_frames = len(CSIDataset(root, chosen, ids, False))
    expected_meta = pretrain_metadata(cfg, manifest, source, label, select, test, chosen, pre_frames)
    if (state.get('method') != 'WAC-Memory-Physics-Pretrain' or state.get('stage') != 'pretrain'
            or state.get('implementation_version') != IMPLEMENTATION_VERSION
            or state.get('run_type') != run_type or state.get('pretrain_manifest') != expected_meta
            or premeta != expected_meta or state.get('config_fingerprint') != pretraining_fingerprint(cfg)):
        raise RuntimeError('pretraining artifact mismatch: rerun v2 pretraining with this config')
    train = CSIDataset(root, label, ids, True)
    select_ds = CSIDataset(root, select, ids, True)
    # No target dataset or GT is opened until after best-checkpoint selection.
    device = torch.device(e.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    base = PhysicalEncoder(int(mc.get('dim', 128))).to(device).eval()
    base.load_state_dict(state['encoder_state_dict'], strict=True)
    workers, bs = int(e.get('num_workers', 0)), int(e.get('batch_size', 64))
    mem_loader = DataLoader(train, int(e.get('memory_batch_size', 128)), shuffle=False,
                            num_workers=workers, collate_fn=collate)
    keys, coords, desc, sequences = [], [], [], []
    print(f'[PHYS-FIN][MEMORY] encoding {len(train)} labeled source frames on {device}', flush=True)
    with torch.no_grad():
        for x, raw, target, sid in mem_loader:
            _, prop, d = base(x.to(device), raw.to(device))
            keys.append(prop.cpu()); desc.append(d.cpu())
            coords.append(target[:, 0]); sequences.append(sid)
    keys, coords, desc, sequences = map(torch.cat, (keys, coords, desc, sequences))
    if len(sequences.unique()) < 2:
        raise ValueError('leave-one-sequence-out training requires at least two source sequences')
    model = PhysAlignPose(
        int(mc.get('dim', 128)), keys, coords, desc, sequences,
        top_k=int(mc.get('top_k', 8)), max_align=float(mc.get('max_align', .35)),
        temperature=float(mc.get('temperature', .1)), physical_weight=float(mc.get('physical_weight', .3)),
        neighbor_sigma=float(mc.get('neighbor_sigma', .15)), positive_k=int(mc.get('positive_k', 64)),
        retrieval_weight=float(cfg.get('loss', {}).get('retrieval_weight', .1)),
        anchor_weight=float(cfg.get('loss', {}).get('anchor_weight', 1.)))
    model.encoder.load_state_dict(base.cpu().state_dict())
    del base, state
    model.configure_finetuning()
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(e.get('learning_rate', 3e-4)), weight_decay=float(e.get('weight_decay', 1e-4)))
    epochs = int(e.get('epochs', 25))
    if epochs < 1:
        raise ValueError('epochs must be positive')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=float(e.get('min_learning_rate', 1e-5)))
    amp = bool(e.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    signature = training_fingerprint(cfg)
    out = resolve_path(cfg['artifacts']['finetune_root'])
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / 'last_state.pth', out / 'best_absolute.pth'
    meta = dict(method='WAC-Memory-Physics', implementation_version=IMPLEMENTATION_VERSION,
                stage='finetune', run_type=run_type, config_fingerprint=signature,
                manifest_fingerprint=fingerprint(manifest), pretrain_checkpoint_sha256=pre_sha,
                memory_fingerprint=tensor_fingerprint(keys, coords, desc, sequences))
    loader = DataLoader(train, bs, shuffle=True, num_workers=workers, collate_fn=collate)
    sel_loader = DataLoader(select_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
    rows, start, best = [], 0, float('inf')
    if last_path.is_file() and bool(e.get('resume', True)):
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if any(state.get(k) != v for k, v in meta.items()):
            raise RuntimeError('finetune resume checkpoint/config/pretraining/memory mismatch; use a fresh directory')
        if not best_path.is_file() or state.get('best_checkpoint_sha256') != file_fingerprint(best_path):
            raise RuntimeError('best finetune checkpoint missing or changed')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        restore_rng(state['rng_state'])
        rows, start, best = list(state['metrics']), int(state['epoch']), float(state['best_metric'])
        print(f'[PHYS-FIN][RESUME] epoch={start} best_select_abs={best:.2f}mm', flush=True)
    elif last_path.exists() or best_path.exists():
        raise RuntimeError('finetune output already contains checkpoints; use resume or a fresh directory')
    write_json(out / 'memory_manifest.json', {**meta, 'memory_count': len(coords),
               'source_only': True, 'pretrain_manifest': premeta})
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'[PHYS-FIN][START] version=2 run={run_type} device={device} amp={amp} '
          f'train={len(train)} select={len(select_ds)} batches={len(loader)} '
          f'trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}', flush=True)
    for epoch in range(start, epochs):
        model.train()
        totals, frames, updates, batches = {}, 0, 0, 0
        begin = time.perf_counter()
        for i, (x, raw, target, sid) in enumerate(loader, 1):
            x, raw, target, sid = x.to(device), raw.to(device), target.to(device), sid.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(x, raw, exclude_seq=sid)
                loss, logs = model.loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite finetune loss at epoch={epoch + 1} batch={i}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(e.get('grad_clip', 1.)), error_if_nonfinite=not amp)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(not amp or scaler.get_scale() >= before)
            n = len(target)
            frames += n
            batches = i
            for k, v in logs.items():
                totals[k] = totals.get(k, 0.) + v * n
            interval = int(e.get('log_interval', 50))
            if interval > 0 and (i == 1 or i % interval == 0):
                print(f'[PHYS-FIN][{epoch + 1}/{epochs}][{i}/{len(loader)}] '
                      f'loss={logs["loss"]:.4f} retrieval_kl={logs["retrieval"]:.4f} '
                      f'root={logs["root"] * 1000:.2f}mm updates={updates}', flush=True)
            if e.get('max_train_batches', 0) and i >= int(e['max_train_batches']):
                break
        if not updates:
            raise RuntimeError('finetuning epoch had no optimizer updates')
        if not e.get('max_train_batches', 0) and (frames != len(train) or batches != len(loader)):
            raise RuntimeError('incomplete fine-tuning epoch')
        scheduler.step()
        val = evaluate(model, sel_loader, device, int(e.get('max_select_batches', 0)))
        row = dict(epoch=epoch + 1, **{f'train_{k}': v / frames for k, v in totals.items()},
                   learning_rate=optimizer.param_groups[0]['lr'], batches=batches, frames_processed=frames,
                   optimizer_updates=updates, skipped_updates=batches - updates,
                   time_s=time.perf_counter() - begin, **val)
        rows.append(row)
        if val['absolute_mpjpe_mm'] < best:
            best = val['absolute_mpjpe_mm']
            atomic_save({**meta, 'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'select_metrics': row}, best_path)
        atomic_save({**meta, 'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'scaler_state_dict': scaler.state_dict(), 'best_metric': best, 'metrics': list(rows),
                     'best_checkpoint_sha256': file_fingerprint(best_path), 'rng_state': rng_state()}, last_path)
        write_json(out / 'metrics.json', rows)
        print(f'[PHYS-FIN][EPOCH {epoch + 1}/{epochs}] loss={row["train_loss"]:.4f} '
              f'kl={row["train_retrieval"]:.4f} abs={val["absolute_mpjpe_mm"]:.2f}mm '
              f'pa={val["pa_mpjpe_mm"]:.2f}mm root={val["root_mae_mm"]:.2f}mm '
              f'anchor={val["anchor_error_mm"]:.2f}mm best={best:.2f}mm '
              f'frames={frames} batches={batches}/{len(loader)} updates={updates} '
              f'lr={row["learning_rate"]:.3g} time={row["time_s"]:.1f}s', flush=True)
    selected = torch.load(best_path, map_location='cpu', weights_only=False)
    if any(selected.get(k) != v for k, v in meta.items()):
        raise RuntimeError('selected checkpoint failed provenance validation')
    model.load_state_dict(selected['model_state_dict'])
    report = dict(**meta, pretrain_source_only=True, pretrain_fraction=premeta['pretrain_fraction'],
                  memory_source_only=True, leave_one_sequence_out=True, test_used_for_selection=False,
                  test_labels_used_for_training_or_selection=False, target_shift_correction=False,
                  best_checkpoint_sha256=file_fingerprint(best_path), select=selected['select_metrics'])
    if run_type == 'formal':
        print('[PHYS-FIN][TEST] evaluating selected checkpoint on full E04', flush=True)
        test_ds = CSIDataset(root, test, ids, True)
        test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=workers, collate_fn=collate)
        report['test'] = evaluate(model, test_loader, device)
        report['test_gt_loaded_only_final_eval'] = True
        print(f'[PHYS-FIN][DONE] best_select_abs={best:.2f}mm '
              f'test_abs={report["test"]["absolute_mpjpe_mm"]:.2f}mm '
              f'test_pa={report["test"]["pa_mpjpe_mm"]:.2f}mm output={out}', flush=True)
    else:
        report['test'] = None
        print(f'[PHYS-FIN][SMOKE-DONE] best_select_abs={best:.2f}mm target_test_skipped=True output={out}', flush=True)
    write_json(out / 'final_report.json', report)


if __name__ == '__main__':
    main()
