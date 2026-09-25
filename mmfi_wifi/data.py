"""
mmfi_wifi.data — MM-Fi 数据集加载 (vendor + 修正)
===================================================
来源: MMFi_dataset-main/mmfi_lib/mmfi.py

相对原版的修正:
  1. wifi-csi 优先读 .npy 缓存 (~0.05ms/帧), 不存在时回退 scipy .mat (~5ms/帧)。
     预处理 (inf->NaN->列均值插值->min-max 归一化) 与原版数学等价,
     增加 max==min 除零保护。
  2. data_list 磁盘缓存: 原版每次启动对 32 万帧做 os.path.getsize 扫描
     (5-15 分钟)。缓存键 = MD5(data_root + data_unit + modality + split +
     全部 subject:action 列表)。旧版 fast_loader 仅以 subject 数量做键,
     不同划分 subject 数相同时会静默串数据 —— 已修复。
  3. 缓存目录固定在 tyr1/.cache/, 不污染数据集目录。

数据格式:
  CSI : wifi-csi/frameXXX.(mat|npy), CSIamp (3, 114, 10) — 3天线 x 114子载波 x 10时间采样
  GT  : ground_truth.npy (297, 17, 3) — 17 关键点 3D 坐标 (x, y, z), 相机坐标系, 单位米
        注意: 第 3 维是 z (深度), 不是 confidence。
"""

import os
import glob
import hashlib
import pickle
import copy
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')

ALL_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
ALL_ACTIONS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
               'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
               'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
DAILY_ACTIONS = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20',
                 'A21', 'A22', 'A23', 'A27']
REHAB_ACTIONS = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16',
                 'A24', 'A25', 'A26']


def scene_for_subject(subject):
    """Return MM-Fi's fixed scene for an official subject identifier."""

    try:
        subject_index = ALL_SUBJECTS.index(subject)
    except ValueError as error:
        raise ValueError(f"Subject does not exist in this dataset: {subject!r}") from error
    return f"E{subject_index // 10 + 1:02d}"


def decode_config(config):
    """与 mmfi_lib.mmfi.decode_config 逻辑一致。"""
    train_form = {}
    val_form = {}
    # 动作范围 (protocol)
    if config['protocol'] == 'protocol1':
        actions = DAILY_ACTIONS
    elif config['protocol'] == 'protocol2':
        actions = REHAB_ACTIONS
    elif config['protocol'] == 'protocol3':
        actions = ALL_ACTIONS
    else:
        raise ValueError(f"未知 protocol: {config['protocol']!r}")
    # 划分方式 (split)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            # 与官方结果保持相同序列，但用局部 RNG，避免污染全局 NumPy 状态。
            idx = np.random.RandomState(rs).permutation(len(ALL_SUBJECTS))
            idx_train = idx[:int(np.floor(ratio * len(ALL_SUBJECTS)))]
            idx_val = idx[int(np.floor(ratio * len(ALL_SUBJECTS))):]
            subjects_train = np.array(ALL_SUBJECTS)[idx_train].tolist()
            subjects_val = np.array(ALL_SUBJECTS)[idx_val].tolist()
            for subject in ALL_SUBJECTS:
                if subject in subjects_train:
                    train_form.setdefault(subject, []).append(action)
                if subject in subjects_val:
                    val_form.setdefault(subject, []).append(action)
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        # 注意: mmfi 官方实现此处硬编码 S01-S30 (E01-E03) / S31-S40 (E04),
        # config 中 cross_scene_split 的 scenes/actions 键不会被读取 (仅作文档)。
        subjects_train = ALL_SUBJECTS[:30]
        subjects_val = ALL_SUBJECTS[30:]
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'manual_split':
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        # manual 仍必须服从 protocol；配置可给全集，实际取交集。
        actions_train = [a for a in config['manual_split']['train_dataset']['actions']
                         if a in actions]
        actions_val = [a for a in config['manual_split']['val_dataset']['actions']
                       if a in actions]
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val
    else:
        raise ValueError(f"未知 split_to_use: {config['split_to_use']!r}")

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form},
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config


class MMFi_Database:
    def __init__(self, data_root):
        # 缓存内路径必须与调用 cwd 无关。
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            scene_path = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_path):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(scene_path)):
                if subject.startswith("."):
                    continue
                subject_path = os.path.join(scene_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(subject_path)):
                    if action.startswith("."):
                        continue
                    action_path = os.path.join(subject_path, action)
                    if not os.path.isdir(action_path):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(action_path, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


def _preprocess_csi(data_mat):
    """
    CSI 预处理, 与 mmfi_lib 原版逻辑数学等价:
      inf -> NaN -> 每个时间片 NaN 用该片均值填充 -> min-max 归一化到 [0,1]
    增加 max==min 除零保护。输入 (3, 114, 10), 输出 float32。
    """
    data_mat = data_mat.astype(np.float64)
    data_mat[np.isinf(data_mat)] = np.nan
    for i in range(data_mat.shape[2]):
        temp_col = data_mat[:, :, i]
        nan_mask = np.isnan(temp_col)
        if nan_mask.any():
            valid = temp_col[~nan_mask]
            temp_col[nan_mask] = valid.mean() if valid.size > 0 else 0.0
    dmin, dmax = np.min(data_mat), np.max(data_mat)
    if dmax - dmin > 1e-8:
        data_mat = (data_mat - dmin) / (dmax - dmin)
    else:
        data_mat = data_mat - dmin
    return np.asarray(data_mat, dtype=np.float32)


def _load_csi_frame(frame_path):
    """读单帧 CSI。优先 .npy, 回退 .mat。返回 (3, 114, 10) float32。"""
    npy_path = frame_path[:-4] + '.npy' if frame_path.endswith('.mat') else frame_path + '.npy'
    if os.path.exists(npy_path):
        return _preprocess_csi(np.load(npy_path))
    import scipy.io as scio
    return _preprocess_csi(scio.loadmat(frame_path)['CSIamp'])


# ---- 打包 CSI 快读 (mmap + per-worker 缓存) ----
# build_packed_cache.py 将每序列 297 帧打包为一个 .npy (float16, 已预处理)
# 这里用 mmap_mode='r' 映射, OS 管理页缓存, 仅读取所需帧的字节
_mmap_cache = {}

# Each DataLoader worker is a separate process, so this cache is process-local.
# Bounding it avoids retaining one mmap/file handle for every subject-action pair.
_GT_CACHE_MAXSIZE = 256
_gt_cache = OrderedDict()


def _load_ground_truth(gt_path):
    """Load a ground-truth array through a process-local bounded LRU mmap cache."""
    arr = _gt_cache.pop(gt_path, None)
    if arr is None:
        arr = np.load(gt_path, mmap_mode='r')
    _gt_cache[gt_path] = arr
    if len(_gt_cache) > _GT_CACHE_MAXSIZE:
        _gt_cache.popitem(last=False)
    return arr


def _load_packed_csi(packed_path, idx):
    """从打包文件读单帧。packed_path: .../wifi-csi-packed.npy, idx: 0-296。
    返回 (3, 114, 10) float32 (float16→float32 无损转换, 数据已预处理)。
    """
    arr = _mmap_cache.get(packed_path)
    if arr is None:
        arr = np.load(packed_path, mmap_mode='r')
        _mmap_cache[packed_path] = arr
    return np.array(arr[idx], dtype=np.float32)


class MMFi_Dataset(Dataset):
    def __init__(self, data_base, data_unit, modality, split, data_form,
                 preload_packed_csi=False, preload_in_workers=False):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()
        # 可选: 将 packed CSI 全量预加载入内存 (可回滚; 默认关保兼容)。
        # preload_packed_csi: 主进程预加载 → __getitem__ 直接读内存 (快)。
        #   ⚠ Windows spawn 会 pickle 整个 Dataset (含预加载数据) 到每个 worker
        #   → num_workers × ~2GB 内存爆炸。仅在 Linux fork 或 num_workers=0 时安全。
        # preload_in_workers: 主进程不预加载 (轻量→spawn 安全), 每个 worker 启动时
        #   各自 _preload_packed_csi() → 每 worker ~2GB, 总 ≈ num_workers × 2GB。
        #   仅当系统 RAM 充足时启用。
        self.preload_packed_csi = preload_packed_csi or preload_in_workers
        self._preloaded_csi = {}
        self._preload_in_workers = preload_in_workers
        if preload_packed_csi and data_unit == 'frame':
            self._preload_packed_csi()

    def get_scene(self, subject):
        return scene_for_subject(subject)

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    # ---- data_list 磁盘缓存 (修正版: 键含完整 subject:action 内容) ----
    def _cache_path(self):
        form_str = ';'.join(
            f"{s}:{','.join(sorted(acts))}"
            for s, acts in sorted(self.data_source.items()))
        key = '|'.join([
            'v3-absolute-paths',
            os.path.abspath(self.data_base.data_root),
            self.data_unit,
            '_'.join(sorted(self.modality)),
            self.split,
            hashlib.md5(form_str.encode('utf-8')).hexdigest(),
        ])
        h = hashlib.md5(key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(_CACHE_DIR, f"datalist_{h}.pkl")

    def load_data(self):
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        data_info = self._scan_data()

        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp_path = cache_path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(data_info, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(f"  [data] data_list 缓存已保存: {cache_path} ({len(data_info):,} 样本)", flush=True)
        except Exception as e:
            print(f"  [data] 缓存保存失败 ({e}), 不影响运行", flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return data_info

    def _scan_data(self):
        print(f"  [data] 扫描数据集 ({self.split}, {len(self.data_source)} subjects, 一次性)...", flush=True)
        data_info = []
        for subject, actions in self.data_source.items():
            scene = self.get_scene(subject)
            for action in actions:
                gt_path = os.path.join(self.data_base.data_root, scene, subject,
                                       action, 'ground_truth.npy')
                if self.data_unit == 'sequence':
                    data_dict = {'modality': self.modality, 'scene': scene,
                                 'subject': subject, 'action': action, 'gt_path': gt_path}
                    for mod in self.modality:
                        data_dict[mod + '_path'] = os.path.join(self.data_base.data_root,
                                                                scene, subject, action, mod)
                    data_info.append(data_dict)
                elif self.data_unit == 'frame':
                    for idx in range(297):
                        data_dict = {'modality': self.modality, 'scene': scene,
                                     'subject': subject, 'action': action,
                                     'gt_path': gt_path, 'idx': idx}
                        data_valid = True
                        for mod in self.modality:
                            mod_path = os.path.join(self.data_base.data_root, scene, subject,
                                                    action, mod,
                                                    "frame{:03d}".format(idx + 1) + self.get_data_type(mod))
                            data_dict[mod + '_path'] = mod_path
                            if os.path.getsize(mod_path) == 0:
                                data_valid = False
                        if data_valid:
                            data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
        print(f"  [data] 扫描完成: {len(data_info):,} 样本", flush=True)
        return data_info

    def _preload_packed_csi(self):
        """将 data_list 中每个唯一的 wifi-csi-packed.npy 全量读入内存。

        仅在 wifi-csi 且 packed 文件存在时生效; 带异常保护与进度打印。
        预加载后 __getitem__ 直接从内存取帧, 避开 mmap 页读开销。
        """
        if 'wifi-csi' not in self.modality:
            return
        packed_paths = []
        seen = set()
        for item in self.data_list:
            data_path = item.get('wifi-csi_path')
            if not data_path:
                continue
            packed_path = os.path.join(os.path.dirname(data_path),
                                       'wifi-csi-packed.npy')
            if packed_path not in seen:
                seen.add(packed_path)
                packed_paths.append(packed_path)
        total = len(packed_paths)
        loaded, failed = 0, 0
        for i, packed_path in enumerate(packed_paths):
            if not os.path.exists(packed_path):
                continue
            try:
                # mmap_mode=None -> 全量读入内存
                self._preloaded_csi[packed_path] = np.load(packed_path, mmap_mode=None)
                loaded += 1
            except Exception as e:
                failed += 1
                print(f"  [data] 预加载失败 {packed_path}: {e}", flush=True)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [data] packed CSI 预加载 {i + 1}/{total} "
                      f"(成功 {loaded}, 失败 {failed})", flush=True)
        print(f"  [data] packed CSI 预加载完成: {loaded}/{total} 序列 (失败 {failed})",
              flush=True)

    # ---- 序列级读取 (本项目未用, 保留完整性) ----
    def read_dir(self, dir_path):
        _, mod = os.path.split(dir_path)
        data = []
        if mod in ['infra1', 'infra2', 'rgb']:
            for arr_file in sorted(glob.glob(os.path.join(dir_path, "frame*.npy"))):
                data.append(np.load(arr_file))
            data = np.array(data)
        elif mod == 'depth':
            import cv2
            for img in sorted(glob.glob(os.path.join(dir_path, "frame*.png"))):
                data.append(cv2.imread(img, cv2.IMREAD_UNCHANGED) * 0.001)
            data = np.array(data)
        elif mod == 'lidar':
            for bin_file in sorted(glob.glob(os.path.join(dir_path, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    data.append(np.frombuffer(f.read(), dtype=np.float64).reshape(-1, 3))
        elif mod == 'mmwave':
            for bin_file in sorted(glob.glob(os.path.join(dir_path, "frame*.bin"))):
                with open(bin_file, 'rb') as f:
                    data.append(np.frombuffer(f.read(), dtype=np.float64).copy().reshape(-1, 5))
        elif mod == 'wifi-csi':
            for mat_file in sorted(glob.glob(os.path.join(dir_path, "frame*.mat"))):
                data.append(_load_csi_frame(mat_file))
            data = np.array(data)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def read_frame(self, frame):
        _mod_dir, _ = os.path.split(frame)
        _, mod = os.path.split(_mod_dir)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            import cv2
            data = cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float64).reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float64).copy().reshape(-1, 5)
        elif mod == 'wifi-csi':
            data = _load_csi_frame(frame)
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = _load_ground_truth(item['gt_path'])

        if self.data_unit == 'sequence':
            # mmap is read-only; copy the returned sample to preserve prior tensor semantics.
            gt_torch = torch.from_numpy(np.array(gt_numpy, copy=True))
            sample = {'modality': item['modality'], 'scene': item['scene'],
                      'subject': item['subject'], 'action': item['action'],
                      'output': gt_torch}
            for mod in item['modality']:
                data_path = item[mod + '_path']
                if os.path.isdir(data_path):
                    data_mod = self.read_dir(data_path)
                else:
                    data_mod = np.load(data_path + '.npy')
                sample['input_' + mod] = data_mod
        elif self.data_unit == 'frame':
            gt_torch = torch.from_numpy(np.array(gt_numpy[item['idx']], copy=True))
            sample = {'modality': item['modality'], 'scene': item['scene'],
                      'subject': item['subject'], 'action': item['action'],
                      'idx': item['idx'],
                      'output': gt_torch}
            for mod in item['modality']:
                data_path = item[mod + '_path']
                # wifi-csi: 优先读打包文件 (297× 更少文件打开, 已预处理)
                if mod == 'wifi-csi':
                    packed_path = os.path.join(os.path.dirname(data_path),
                                               'wifi-csi-packed.npy')
                    # 命中预加载缓存 -> 直接从内存取帧 (float16->float32 无损)
                    if self.preload_packed_csi and packed_path in self._preloaded_csi:
                        sample['input_' + mod] = np.array(
                            self._preloaded_csi[packed_path][item['idx']],
                            dtype=np.float32)
                        continue
                    if os.path.exists(packed_path):
                        sample['input_' + mod] = _load_packed_csi(
                            packed_path, item['idx'])
                        continue
                if os.path.isfile(data_path):
                    sample['input_' + mod] = self.read_frame(data_path)
                else:
                    raise ValueError('{} is not a file!'.format(data_path))
        else:
            raise ValueError('Unsupport data unit!')
        return sample


def make_dataset(dataset_root, config, preload_packed_csi=False,
                 preload_in_workers=False):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
    train_dataset = MMFi_Dataset(database, config['data_unit'],
                                 preload_packed_csi=preload_packed_csi,
                                 preload_in_workers=preload_in_workers,
                                 **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, config['data_unit'],
                               preload_packed_csi=preload_packed_csi,
                               preload_in_workers=preload_in_workers,
                               **config_dataset['val_dataset'])
    return train_dataset, val_dataset


def make_manual_test_dataset(dataset_root, config):
    """为 manual_split 构造 config.test_subjects 指定的真正 held-out 测试集。"""
    if config.get('split_to_use') != 'manual_split':
        raise ValueError("make_manual_test_dataset 仅适用于 manual_split")
    test_subjects = config.get('test_subjects')
    if not test_subjects:
        raise ValueError("manual_split 缺少 test_subjects")
    test_config = copy.deepcopy(config)
    test_config['manual_split']['train_dataset']['subjects'] = []
    test_config['manual_split']['val_dataset']['subjects'] = list(test_subjects)
    _, test_dataset = make_dataset(dataset_root, test_config)
    return test_dataset


def collate_fn_padd(batch):
    """与 mmfi_lib 原版一致。"""
    batch_data = {'modality': batch[0]['modality'],
                  'scene': [sample['scene'] for sample in batch],
                  'subject': [sample['subject'] for sample in batch],
                  'action': [sample['action'] for sample in batch],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None}
    _output = [np.array(sample['output']) for sample in batch]
    batch_data['output'] = torch.FloatTensor(np.array(_output))

    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 0, 2)
            batch_data['input_' + mod] = _input
        else:
            _input = [np.array(sample['input_' + mod]) for sample in batch]
            batch_data['input_' + mod] = torch.FloatTensor(np.array(_input))

    return batch_data


def _preload_worker_init(worker_id):
    """DataLoader worker 启动钩子: 若 Dataset 标记了 preload_in_workers,
    则在 worker 进程内各自执行 _preload_packed_csi()。
    主进程 Dataset 不持有大数据 → pickle 轻量 → spawn 不爆内存。
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    # Subset (来自序列级划分) 包裹原始 Dataset, 需穿透取底层
    while hasattr(ds, 'dataset'):
        ds = ds.dataset
    if getattr(ds, '_preload_in_workers', False):
        ds._preload_packed_csi()


def make_dataloader(dataset, is_training, generator, batch_size, num_workers=8,
                    prefetch_factor=4):
    """带 pin_memory / prefetch / persistent_workers 的 DataLoader。"""
    kwargs = dict(batch_size=batch_size, collate_fn=collate_fn_padd,
                  shuffle=is_training, drop_last=is_training,
                  generator=generator, pin_memory=True, num_workers=num_workers)
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = True
        kwargs['worker_init_fn'] = _preload_worker_init
    return DataLoader(dataset, **kwargs)
