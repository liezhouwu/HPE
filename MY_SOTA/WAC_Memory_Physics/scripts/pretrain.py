from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from MY_SOTA.WAC_Memory_Physics.src.model import IMPLEMENTATION_VERSION, PretrainModel, pretrain_target
from MY_SOTA.WAC_Memory_Physics.scripts.common import *


def augment(x, raw, settings):
    b, device = len(x), x.device
    link_drop = float(settings.get('link_drop', .1))
    band_drop = float(settings.get('band_drop', .15))
    gain_range = float(settings.get('gain_range', .03))
    offset_range = float(settings.get('offset_range', .005))
    if not 0 <= link_drop < 1 or not 0 <= band_drop < 1 or gain_range < 0 or offset_range < 0:
        raise ValueError('invalid pretraining augmentation')
    mask = torch.rand(b, 3, 1, 1, device=device) >= link_drop
    # Retain at least one link for every sample.
    empty = ~mask.flatten(1).any(1)
    mask[empty, 0] = True
    bands = torch.rand(b, 3, 6, 1, device=device) >= band_drop
    bands[:, :, 0] = True
    bands = F.interpolate(bands.float(), size=(114, 1), mode='nearest')
    mask = mask.float() * bands
    gain = 1 + torch.empty(b, 1, 1, 1, device=device).uniform_(-gain_range, gain_range)
    offset = torch.empty(b, 1, 1, 1, device=device).uniform_(-offset_range, offset_range)
    # Global gain need not alter min-max normalized CSI. Keep raw amplitude
    # changes small so absolute location cues are not aggressively erased.
    xa = x * mask
    ra = (raw * gain + offset).clamp_min(0) * mask
    nuisance = torch.stack([
        (gain.flatten() - 1) / max(gain_range, 1e-8),
        offset.flatten() / max(offset_range, 1e-8),
        1 - mask.mean((1, 2, 3)),
    ], dim=-1)
    return xa, ra, nuisance


@torch.no_grad()
def fit_target_stats(model, dataset, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    total = torch.zeros_like(model.target_mean, dtype=torch.float64, device=device)
    squares, n = total.clone(), 0
    for _, raw, _ in loader:
        y = pretrain_target(raw.to(device)).double()
        total += y.sum(0)
        squares += y.square().sum(0)
        n += len(raw)
    mean = total / n
    std = (squares / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    model.target_mean.copy_(mean.float())
    model.target_std.copy_(std.float())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    e = cfg['experiment']
    run_type = validate_run_paths(cfg)
    seed_all(int(e['seed']))
    manifest, source, label, select, test = load_protocol(cfg)
    chosen = pretrain_selection(cfg, source, label, select, test)
    ids = {seq_string(k): i for i, k in enumerate(sorted(source | select | test))}
    dataset = CSIDataset(resolve_path(e['dataset_root']), chosen, ids, False)
    if len(dataset) < 2:
        raise ValueError('pretraining needs at least two CSI frames')
    batch_size, workers = int(e.get('batch_size', 64)), int(e.get('num_workers', 0))
    if batch_size < 2:
        raise ValueError('pretraining batch_size must be at least two')
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=workers, collate_fn=collate)
    out = resolve_path(cfg['artifacts']['pretrain_root'])
    out.mkdir(parents=True, exist_ok=True)
    best_path = resolve_path(cfg['artifacts']['pretrain_checkpoint'])
    last_path = out / 'pretrain_last.pth'
    meta = pretrain_metadata(cfg, manifest, source, label, select, test, chosen, len(dataset))
    signature = pretraining_fingerprint(cfg)
    device = torch.device(e.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = PretrainModel(int(cfg['model'].get('dim', 128))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(e.get('learning_rate', 3e-4)),
                                  weight_decay=float(e.get('weight_decay', 1e-4)))
    epochs = int(e.get('pretrain_epochs', 20))
    if epochs < 1:
        raise ValueError('pretrain_epochs must be positive')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=float(e.get('min_learning_rate', 1e-5)))
    amp = bool(e.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    rows, start, best = [], 0, float('inf')
    if last_path.is_file() and bool(e.get('resume', True)):
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if (state.get('implementation_version') != IMPLEMENTATION_VERSION
                or state.get('method') != 'WAC-Memory-Physics-Pretrain'
                or state.get('stage') != 'pretrain'
                or state.get('run_type') != run_type
                or state.get('config_fingerprint') != signature
                or state.get('pretrain_manifest') != meta):
            raise RuntimeError('pretrain resume mismatch; use a fresh output directory')
        if not best_path.is_file() or state.get('best_checkpoint_sha256') != file_fingerprint(best_path):
            raise RuntimeError('best pretrain checkpoint missing or changed')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        restore_rng(state['rng_state'])
        start, best, rows = int(state['epoch']), float(state['best_loss']), list(state['metrics'])
        print(f'[PHYS-PRE][RESUME] epoch={start} best_loss={best:.5f}', flush=True)
    else:
        if last_path.exists() or best_path.exists():
            raise RuntimeError('pretrain output already contains checkpoints; use resume or a fresh directory')
        fit_target_stats(model, dataset, batch_size, workers, device)
    write_json(out / 'pretrain_manifest.json', meta)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'[PHYS-PRE][START] version=2 run={run_type} device={device} amp={amp} '
          f'keys={len(chosen)} frames={len(dataset)} batches={len(loader)} fraction={meta["pretrain_fraction"]}', flush=True)
    settings = cfg.get('pretraining', {})
    for epoch in range(start, epochs):
        model.train()
        totals, frames, updates, batches = {}, 0, 0, 0
        begin = time.perf_counter()
        for i, (x, raw, _) in enumerate(loader, 1):
            x, raw = x.to(device), raw.to(device)
            optimizer.zero_grad(set_to_none=True)  # One independent update per batch.
            xa, ra, nuisance_target = augment(x, raw, settings)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred, nuisance, body_aug = model(xa, ra)
                with torch.no_grad():
                    body_full = model(x, raw)[2]
                    target = (pretrain_target(raw) - model.target_mean) / model.target_std
                reconstruction = F.smooth_l1_loss(pred.float(), target)
                nuisance_loss = F.smooth_l1_loss(nuisance.float(), nuisance_target)
                consistency = 1 - F.cosine_similarity(body_aug.float(), body_full.float(), dim=-1).mean()
                # Explicit non-collapse signal. A single stop-gradient cosine
                # target admits a constant representation as a trivial optimum.
                variance = F.relu(.5 - (body_aug.float().var(0, correction=0) + 1e-4).sqrt()).mean()
                loss = (reconstruction + float(settings.get('nuisance_weight', .05)) * nuisance_loss
                        + float(settings.get('consistency_weight', .02)) * consistency
                        + float(settings.get('variance_weight', .1)) * variance)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite pretrain loss at epoch={epoch + 1} batch={i}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(e.get('grad_clip', 1.)), error_if_nonfinite=not amp)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(not amp or scaler.get_scale() >= before)
            n = len(x)
            frames += n
            batches = i
            logs = dict(loss=loss, reconstruction_loss=reconstruction, nuisance_loss=nuisance_loss,
                        consistency_loss=consistency, variance_loss=variance)
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * n
            interval = int(e.get('log_interval', 50))
            if interval > 0 and (i == 1 or i % interval == 0):
                print(f'[PHYS-PRE][{epoch + 1}/{epochs}][{i}/{len(loader)}] '
                      f'loss={float(loss.detach()):.5f} rec={float(reconstruction.detach()):.5f} updates={updates}', flush=True)
            if e.get('max_pretrain_batches', 0) and i >= int(e['max_pretrain_batches']):
                break
        if not updates:
            raise RuntimeError('pretraining epoch had no optimizer updates')
        if not e.get('max_pretrain_batches', 0) and (frames != len(dataset) or batches != len(loader)):
            raise RuntimeError('incomplete pretraining epoch')
        scheduler.step()
        row = {'epoch': epoch + 1, **{k: v / frames for k, v in totals.items()},
               'learning_rate': optimizer.param_groups[0]['lr'], 'batches': batches,
               'frames_processed': frames, 'optimizer_updates': updates, 'skipped_updates': batches - updates,
               'time_s': time.perf_counter() - begin}
        rows.append(row)
        payload = dict(method='WAC-Memory-Physics-Pretrain', stage='pretrain',
                       implementation_version=IMPLEMENTATION_VERSION, run_type=run_type,
                       epoch=epoch + 1, model_state_dict=model.state_dict(),
                       encoder_state_dict=model.encoder.state_dict(), pretrain_manifest=meta,
                       config_fingerprint=signature, metrics=list(rows))
        if row['loss'] < best:
            best = row['loss']
            # Serialize immediately: state_dict is a view of live parameters.
            atomic_save(payload, best_path)
        atomic_save({**payload, 'best_loss': best, 'best_checkpoint_sha256': file_fingerprint(best_path),
                     'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                     'scaler_state_dict': scaler.state_dict(), 'rng_state': rng_state()}, last_path)
        write_json(out / 'metrics.json', rows)
        print(f'[PHYS-PRE][EPOCH {epoch + 1}/{epochs}] loss={row["loss"]:.5f} '
              f'rec={row["reconstruction_loss"]:.5f} nuisance={row["nuisance_loss"]:.5f} '
              f'cons={row["consistency_loss"]:.5f} var={row["variance_loss"]:.5f} '
              f'batches={batches}/{len(loader)} frames={frames} updates={updates} '
              f'best={best:.5f} time={row["time_s"]:.1f}s', flush=True)
    print(f'[PHYS-PRE][DONE] best_loss={best:.5f} output={out}', flush=True)


if __name__ == '__main__':
    main()
