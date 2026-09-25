"""MM-Fi WiFi-CSI 3D 姿态估计包。

采用延迟导入：单独使用数据、指标或配置工具时，不必先加载 PyTorch 模型。
训练脚本仍可通过原来的公开名称访问模型类。
"""
from importlib import import_module

_MODULES = {name: f"mmfi_wifi.{name}" for name in ("data", "engine", "metrics", "model")}
_EXPORTS = {
    "AxisStats": ("mmfi_wifi.metafi_decoder", "AxisStats"),
    "LEGACY_AXIS_STATS": ("mmfi_wifi.metafi_decoder", "LEGACY_AXIS_STATS"),
    "MetaFiPoseDecoder": ("mmfi_wifi.metafi_decoder", "MetaFiPoseDecoder"),
    "EncoderOutput": ("mmfi_wifi.metafi_encoder", "EncoderOutput"),
    "MetaFiEncoder": ("mmfi_wifi.metafi_encoder", "MetaFiEncoder"),
    "MetaFiPoseModel": ("mmfi_wifi.metafi_pose_model", "MetaFiPoseModel"),
    "GT_AXIS_MEAN": ("mmfi_wifi.model", "GT_AXIS_MEAN"),
    "GT_AXIS_STD": ("mmfi_wifi.model", "GT_AXIS_STD"),
    "posenet": ("mmfi_wifi.model", "posenet"),
    "weights_init": ("mmfi_wifi.model", "weights_init"),
}

def __getattr__(name):
    """按需加载公开对象，保持原有导入路径不变。"""
    if name in _MODULES:
        value = import_module(_MODULES[name])
    elif name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value

__all__ = [*(_MODULES), *(_EXPORTS)]
