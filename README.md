# HPE：基于 MM-Fi WiFi-CSI 的三维人体姿态估计

本仓库保存 MM-Fi WiFi-CSI 三维人体姿态估计项目的核心代码和配置文件，包含监督学习、自监督学习、ViT-MAE，以及针对 MM-Fi 任务适配的 WiFi-JEPA 实现。

仓库不包含原始数据集、模型权重、训练结果和训练日志。

## 快速开始

请在仓库根目录执行以下命令。

### WiFi-JEPA 预训练

```powershell
python WIFIJEPA/scripts/pretrain.py --config WIFIJEPA/configs/matched_small.yaml
```

### 下游微调

```powershell
python WIFIJEPA/scripts/finetune.py --config WIFIJEPA/configs/matched_small.yaml
```

配置项 `experiment.mode` 支持两种模式：

```yaml
experiment:
  mode: supervised
```

表示运行结构化 ViT 的纯监督对照；改为 `mode: jepa` 表示加载 WiFi-JEPA context encoder 后进行姿态估计微调。

当前默认配置为：

```
protocol1 / strict / random_split / seed42 / 2shot
```

切换协议、数据划分或标签量时，只需修改配置，审计文件和结果目录会自动适配。

## WiFi-JEPA 核心代码

```text
WIFIJEPA/
  src/model.py       结构化 tokenizer、ViT、预测器、JEPA 和姿态模型
  src/masking.py     时间-链路网格上的整条链路遮挡
  src/data.py        MM-Fi CSI 读取和归一化消融接口
  src/paths.py       自动解析审计、checkpoint 和结果目录
  scripts/pretrain.py  WiFi-JEPA 潜在特征预训练
  scripts/finetune.py  结构化监督对照或 JEPA 微调
  configs/matched_small.yaml
  tests/test_core.py
```

MM-Fi 输入为幅度 CSI，单帧形状为 `(3,114,10)`，重排为 `(C,T,L)=(114,10,3)`，并转换为 30 个时间-链路 token。

## 项目结构

```text
HPE/
  configs/          监督学习和已有 SSL 配置
  dataset/          空目录占位；本地运行时放置 MM-Fi 数据集
  mmfi_wifi/        数据读取、训练引擎、指标和数据边界
  pose_ssl/         已有自监督模型和预训练工具
  result/           空目录占位；用于本地保存结果
  scripts/          训练、审计和结果处理入口
  WIFIJEPA/         WiFi-JEPA 核心代码和配置
  README.md
  requirements.txt
```

## 数据集结构

`WIFIJEPA/configs/matched_small.yaml` 的默认数据路径为 `dataset`：

```text
dataset/
  E01/S01/A01/
    ground_truth.npy
    wifi-csi/frame001.mat ... frame297.mat
  E02/
  E03/
  E04/
```

CSI 单帧幅度数据形状为 `(3,114,10)`，姿态标签形状为 `(297,17,3)`。

## 结果结构

运行结果不会提交到 GitHub，只保存在本地：

```text
WIFIJEPA/results/pretrain/<protocol>/<split>/seed<seed>/<configuration-id>/
  encoder.pth
  pretrain_manifest.json
  pretrain_metrics.yaml

WIFIJEPA/result_metafi_ssl/<method>/<protocol>/<split>/seed<seed>/<label-budget>/<configuration-id>/
  best_absolute.pth
  metrics.csv
  final_report.json
  done.txt
```

`best_absolute.pth` 和 `test_mpjpe_mm` 是实际关节位置误差的主要输出。

## 安装

建议使用 Python 3.10 或更高版本，并先安装与本机 CUDA 驱动兼容的 PyTorch 和 torchvision：

```powershell
python -m pip install -r requirements.txt
```
