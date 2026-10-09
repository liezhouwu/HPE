from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mmfi_wifi import data as mmfi_data
from mmfi_wifi.data import _load_csi_frame, _load_packed_csi
from pose_ssl.metafi.pretrain_data import MetaFiPretrainDataset


def preprocess_without_minmax(data):
    # Keep NaN/Inf handling while skipping min-max normalization for ablation.
    data = np.asarray(data, dtype=np.float64)
    data[np.isinf(data)] = np.nan
    for index in range(data.shape[2]):
        column = data[:, :, index]
        missing = np.isnan(column)
        if missing.any():
            valid = column[~missing]
            column[missing] = valid.mean() if valid.size else 0.0
    return data.astype(np.float32)


def _load_raw_frame(frame_path: Path):
    import scipy.io as scio

    npy_path = frame_path.with_suffix(".npy")
    raw = np.load(npy_path) if npy_path.is_file() else scio.loadmat(frame_path)["CSIamp"]
    return preprocess_without_minmax(raw)


def _load_raw_packed_frame(packed_path, index: int):
    frame_path = Path(packed_path).parent / f"frame{index + 1:03d}.mat"
    return _load_raw_frame(frame_path)


@contextmanager
def csi_preprocessing(normalize: bool):
    """Temporarily select the CSI normalization path for ablation."""
    if normalize:
        yield
        return
    old_preprocess = mmfi_data._preprocess_csi
    old_packed_loader = mmfi_data._load_packed_csi
    mmfi_data._preprocess_csi = preprocess_without_minmax
    mmfi_data._load_packed_csi = _load_raw_packed_frame
    try:
        yield
    finally:
        mmfi_data._preprocess_csi = old_preprocess
        mmfi_data._load_packed_csi = old_packed_loader


class WiFiJEPAData(MetaFiPretrainDataset):
    """Load CSI within the audited sequence boundary."""

    def __init__(self, *args, normalize: bool = True, **kwargs):
        self.normalize = normalize
        super().__init__(*args, **kwargs)

    def _load_record(self, record):
        if self.normalize and record.packed_path is not None:
            data = _load_packed_csi(str(record.packed_path), record.frame_index)
        elif self.normalize:
            data = _load_csi_frame(str(record.frame_path))
        else:
            data = _load_raw_frame(Path(record.frame_path))
        return torch.as_tensor(data, dtype=torch.float32).contiguous()
