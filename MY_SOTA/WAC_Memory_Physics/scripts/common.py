from __future__ import annotations
import hashlib, json, os, random
from functools import lru_cache
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from mmfi_wifi.data import _load_csi_frame, _load_ground_truth, _preprocess_csi
from mmfi_wifi.sequence_keys import SequenceKey

ROOT = Path(__file__).resolve().parents[3]
CSI_SHAPE = (3, 114, 10)

def fingerprint(x): return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()
def read_keys(path):
    p=json.loads(Path(path).read_text(encoding='utf-8-sig')); return frozenset(SequenceKey(*x) for x in p['selected_keys'])
def raw_csi(path):
    p=Path(path); q=p.with_suffix('.npy'); x=np.load(q) if q.is_file() else __import__('scipy.io').io.loadmat(p)['CSIamp']
    x=np.asarray(x,dtype=np.float32); good=np.isfinite(x); fill=float(x[good].mean()) if good.any() else 0.; x=np.nan_to_num(x,nan=fill,posinf=fill,neginf=fill)
    if tuple(x.shape)!=CSI_SHAPE: raise ValueError(f'bad CSI shape {x.shape} at {path}')
    return x
@lru_cache(maxsize=200000)
def cached_raw(path): return torch.as_tensor(raw_csi(Path(path))/60.,dtype=torch.float32)
def paths(seq):
    p=sorted((seq/'wifi-csi').glob('frame*.mat')); return p or sorted((seq/'wifi-csi').glob('frame*.npy'))
def seq_string(k): return str(k)
class CSIDataset(Dataset):
    def __init__(self, root, keys, seq_to_id, with_gt):
        self.root=Path(root); self.items=[]; self.with_gt=with_gt; self.seq_to_id=seq_to_id
        for k in sorted(keys):
            seq=self.root/k.scene/k.subject/k.action; ps=paths(seq)
            if not ps: raise FileNotFoundError(seq/'wifi-csi')
            gt=None
            if with_gt:
                gt=np.load(seq/'ground_truth.npy',mmap_mode='r')
                if len(gt)!=len(ps) or tuple(gt.shape[1:])!=(17,3): raise ValueError(f'CSI/GT mismatch {k}')
            for i,p in enumerate(ps): self.items.append((k,i,p))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        k,idx,p=self.items[i]
        # The shared MAT loader does not accept an NPY path directly.
        normalized = _preprocess_csi(np.load(p)) if p.suffix == '.npy' else _load_csi_frame(str(p))
        x=torch.as_tensor(normalized,dtype=torch.float32); raw=cached_raw(str(p)); sid=torch.tensor(self.seq_to_id[seq_string(k)],dtype=torch.long)
        if self.with_gt:
            seq=self.root/k.scene/k.subject/k.action; y=torch.as_tensor(_load_ground_truth(str(seq/'ground_truth.npy'))[idx].copy(),dtype=torch.float32); return x,raw,y,sid
        return x,raw,sid
def collate(batch):
    if len(batch[0])==3:
        a,b,c=zip(*batch); return torch.stack(a),torch.stack(b),torch.stack(c)
    a,b,c,d=zip(*batch); return torch.stack(a),torch.stack(b),torch.stack(c),torch.stack(d)
def atomic_save(obj,path):
    p=Path(path); t=p.with_suffix(p.suffix+'.tmp'); torch.save(obj,t); os.replace(t,p)
def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def resolve_path(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def training_fingerprint(cfg):
    # Resuming and console verbosity do not alter optimizer trajectories.
    cfg = json.loads(json.dumps(cfg))
    for field in ('resume', 'log_interval'):
        cfg['experiment'].pop(field, None)
    return fingerprint(cfg)

def pretraining_fingerprint(cfg):
    # Fine-tuning-only settings may change without invalidating SSL weights.
    names = ('dataset_root', 'manifest', 'label_manifest', 'protocol', 'split', 'scope',
             'seed', 'pretrain_fraction', 'pretrain_epochs', 'batch_size', 'num_workers',
             'learning_rate', 'weight_decay', 'min_learning_rate', 'amp', 'device',
             'grad_clip', 'max_pretrain_batches', 'run_type')
    return fingerprint({'version': 2, 'experiment': {k: cfg['experiment'].get(k) for k in names},
                        'dim': cfg['model'].get('dim', 128), 'pretraining': cfg.get('pretraining', {})})

def load_protocol(cfg):
    e = cfg['experiment']
    manifest = json.loads(resolve_path(e['manifest']).read_text(encoding='utf-8-sig'))
    for field in ('protocol', 'split', 'scope', 'seed'):
        if str(e.get(field)) != str(manifest.get(field)):
            raise ValueError(f'config/manifest mismatch: {field}')
    source = frozenset(SequenceKey(*x) for x in manifest['internal_train_keys'])
    select = frozenset(SequenceKey(*x) for x in manifest['select_keys'])
    test = frozenset(SequenceKey(*x) for x in manifest['test_keys'])
    label = read_keys(resolve_path(e['label_manifest']))
    if not source or not label or not select or not test:
        raise ValueError('empty protocol split')
    if not label.issubset(source) or source & (select | test) or select & test:
        raise ValueError('train/select/test sequence overlap')
    if e['split'] == 'cross_scene_split':
        if any(k.scene not in {'E01', 'E02', 'E03'} for k in source | select):
            raise ValueError('source split contains target scene')
        if any(k.scene != 'E04' for k in test):
            raise ValueError('unexpected test scene')
    return manifest, source, label, select, test

def pretrain_selection(cfg, source, label, select, test):
    fraction = float(cfg['experiment'].get('pretrain_fraction', .2))
    if not 0 < fraction <= 1:
        raise ValueError('pretrain_fraction must be in (0,1]')
    candidates = sorted(source - label - select - test)
    if not candidates:
        raise ValueError('empty eligible source CSI pool')
    random.Random(int(cfg['experiment']['seed'])).shuffle(candidates)
    return frozenset(candidates[:max(1, round(len(candidates) * fraction))])

def pretrain_metadata(cfg, manifest, source, label, select, test, chosen, frames):
    return {'implementation_version': 2, 'source_only': True,
            'pretrain_fraction': float(cfg['experiment']['pretrain_fraction']),
            'selection_unit': 'sequence', 'seed': int(cfg['experiment']['seed']),
            'protocol': cfg['experiment']['protocol'], 'split': cfg['experiment']['split'],
            'scope': cfg['experiment']['scope'], 'manifest_fingerprint': fingerprint(manifest),
            'source_fingerprint': fingerprint(sorted(map(str, source))),
            'excluded_label_fingerprint': fingerprint(sorted(map(str, label))),
            'excluded_select_fingerprint': fingerprint(sorted(map(str, select))),
            'excluded_test_fingerprint': fingerprint(sorted(map(str, test))),
            'selected_keys': [[k.scene, k.subject, k.action] for k in sorted(chosen)],
            'selected_key_fingerprint': fingerprint(sorted(map(str, chosen))),
            'candidate_sequences': len(source - label - select - test), 'frames': frames}

def validate_run_paths(cfg):
    e, artifacts = cfg['experiment'], cfg['artifacts']
    run_type = e.get('run_type', 'formal')
    if run_type not in {'formal', 'smoke'}:
        raise ValueError('run_type must be formal or smoke')
    limited = any(int(e.get(k, 0)) for k in
                  ('max_pretrain_batches', 'max_train_batches', 'max_select_batches', 'max_test_batches'))
    if limited and run_type != 'smoke':
        raise ValueError('limited-batch validation requires run_type: smoke')
    pre = resolve_path(artifacts['pretrain_root'])
    ck = resolve_path(artifacts['pretrain_checkpoint'])
    if ck.parent != pre:
        raise ValueError('pretrain_checkpoint must be inside pretrain_root to prevent artifact mixing')
    if run_type == 'smoke':
        for name in ('pretrain_root', 'finetune_root'):
            if name in artifacts and not any(word in str(artifacts[name]).lower() for word in ('smoke', 'diag', 'validation')):
                raise ValueError(f'smoke {name} must use an isolated smoke/diag/validation directory')
    return run_type

def write_json(path, payload):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, p)

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
