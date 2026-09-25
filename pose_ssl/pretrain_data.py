"""
pose_ssl.pretrain_data — SSL 预训练语料 (无标签)
===================================================
语料 = 协议内全部数据 (全部 40 被试 × 4 场景, 含测试分布 —— 已确认的 SSL
标准设定: 无标签数据充裕是 SSL 的前提, 测试被试/场景的 CSI 以无标签形式
参与预训练)。

复用 mmfi_wifi.data 的打包缓存 (mmap float16) 快读; 数据集按帧展开,
每样本返回 (3, 114, 10) float32 张量。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from mmfi_wifi.data import (MMFi_Database, MMFi_Dataset, ALL_SUBJECTS,
                            DAILY_ACTIONS, REHAB_ACTIONS, ALL_ACTIONS)

PROTOCOL_ACTIONS = {
    'protocol1': DAILY_ACTIONS,
    'protocol2': REHAB_ACTIONS,
    'protocol3': ALL_ACTIONS,
}


class PretrainDataset(Dataset):
    """无标签帧级 CSI 数据集。包装 MMFi_Dataset, 只取 CSI 输入。"""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        csi = sample['input_wifi-csi']
        if not isinstance(csi, torch.Tensor):
            csi = torch.from_numpy(np.asarray(csi, dtype=np.float32))
        return csi.float()


def make_pretrain_dataset(dataset_root, protocol):
    """构建协议内全量无标签预训练语料。返回 (PretrainDataset, n_sequences)。"""
    if protocol not in PROTOCOL_ACTIONS:
        raise ValueError(f"未知 protocol: {protocol!r}")
    actions = list(PROTOCOL_ACTIONS[protocol])
    data_form = {subject: actions for subject in ALL_SUBJECTS}
    database = MMFi_Database(dataset_root)
    base = MMFi_Dataset(database, 'frame', 'wifi-csi', 'training', data_form)
    n_sequences = len({(d['scene'], d['subject'], d['action'])
                       for d in base.data_list})
    return PretrainDataset(base), n_sequences
