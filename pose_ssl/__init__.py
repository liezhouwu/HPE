"""姿态估计模型、损失和 MetaFi SSL 方法。"""
from importlib import import_module

_EXPORTS = {
    "PoseModel": ("pose_ssl.model", "PoseModel"),
    "ResNet18Encoder": ("pose_ssl.model", "ResNet18Encoder"),
    "ViTCSIEncoder": ("pose_ssl.model", "ViTCSIEncoder"),
    "PoseHead": ("pose_ssl.model", "PoseHead"),
    "build_pose_model": ("pose_ssl.model", "build_pose_model"),
    "PoseLoss": ("pose_ssl.loss", "PoseLoss"),
    "BoneLoss": ("pose_ssl.loss", "BoneLoss"),
    "COCO_BONES": ("pose_ssl.loss", "COCO_BONES"),
}

def __getattr__(name):
    """按需导入 PyTorch 模型，配置工具无需依赖完整训练环境。"""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = list(_EXPORTS)
