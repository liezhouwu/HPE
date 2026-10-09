# HPE：WiFi-CSI 三维人体姿态估计

本项目使用 MM-Fi 的 WiFi-CSI 数据进行三维人体姿态估计，包含监督基线、MetaFi/ResNet-34 自监督预训练与微调，以及独立的 ViT-MAE 预训练与微调。支持 Protocol 1/2/3 与 Random、Cross-subject、Cross-scene 三种划分。

> 本仓库只提供源码和配置；不提供原始数据、预训练权重、实验结果或训练日志。首次运行需要自行准备 MM-Fi 数据集。

## 核心目录与入口

| 路径 | 作用 |
| --- | --- |
| `mmfi_wifi/` | MM-Fi 数据加载、训练引擎、模型、姿态评价指标、数据边界与标签清单校验、运行身份。 |
| `pose_ssl/metafi/` | SimCLR、MoCo、SwAV、RelPos、MFM、MAE 的 MetaFi 自监督方法、抽样和 checkpoint 管理。 |
| `pose_ssl/vit_ssl/`、`pose_ssl/model.py` | ViT-CSI-small 编码器及 ViT-MAE 模型。 |
| `scripts/run.py`、`scripts/metafi_ssl/` | 监督基线、MetaFi 全量/小样本 SSL 的审计、预训练、监督训练和微调入口。 |
| `scripts/reproduction/` | 监督基线训练、跨划分运行和测试入口。 |
| `scripts/vit_ssl/` | ViT-MAE 审计清单复用、预训练与微调；`run.py` 是一条命令运行三种划分的入口。 |
| `scripts/report_results.py` | 扫描已完成实验并生成本地结果汇总；结果文件不纳入本仓库。 |
| `configs/` | 官方监督基线、MetaFi 全量/小样本、ViT 全量/小样本的 YAML 配置。 |

本仓库未附测试代码和额外说明文档。

## 环境和数据

- Python 3.10+；需要 PyTorch 和 torchvision 版本互相匹配。使用 CUDA 时请安装与本机驱动兼容的 PyTorch 构建；本地开发环境使用 Python 3.12。
- 在仓库根目录执行 `python -m pip install -r requirements.txt`。如需 CUDA，请先正确安装对应的 PyTorch/torchvision，再安装其余依赖。
- 下载并自行放置 MM-Fi 数据集，保留原有场景、受试者与动作层级。本仓库不含数据。
- 五份 `configs/*.yaml` 的 `dataset_root` 默认都是 `../my_dataset`，表示数据集位于此仓库的同级目录；如放在其他位置，请先修改配置。运行 ViT 入口时传入的数据集位置也应与配置一致。
- 下方命令在仓库根目录执行。`4shot` 指每个动作使用四条完整标注序列；其他可用预算以运行入口和配置为准。

## 运行示例

### 官方监督基线

~~~powershell
python scripts/run.py baseline all --protocol protocol3
~~~

### MetaFi 少样本：监督对照和 SSL

~~~powershell
python scripts/run.py small baseline all --protocol protocol3 --label-budget 4shot
python scripts/run.py small swav all --protocol protocol3 --label-budget 4shot
~~~

`swav` 可换成 `simclr`、`moco`、`relpos`、`mfm` 或 `mae`。`all` 依次运行三种划分；只运行一个划分可用 `single --split cross_scene_split`。全量预训练入口示例：`python scripts/run.py ssl simclr all --protocol protocol3 --label-budget 4shot`。`scripts/run.py` 会按需创建数据审计，并在已完成实验后调用结果汇总脚本。

### ViT-MAE 小样本：仅 MAE-ViT，三种划分

ViT 入口**复用 MetaFi 的审计清单**，而不会在空目录自动生成审计。新克隆仓库若尚无对应清单，先对每个划分单独审计（PowerShell 示例）：

~~~powershell
foreach ($split in @("random_split", "cross_subject_split", "cross_scene_split")) {
    $audit = "result_metafi_ssl/runs/protocol3/strict/$split/seed42/audit/b4s"
    python scripts/metafi_ssl/audit_data.py ../my_dataset configs/baseline_config.yaml `
        --protocol protocol3 --split $split --scope strict --seed 42 `
        --label-budget 4shot --output-dir $audit
    if ($LASTEXITCODE -ne 0) { throw "审计失败：$split" }
}
~~~

审计目录已存在且产物完整时，不要重新运行审计命令覆盖原目录。然后执行：

~~~powershell
python scripts/vit_ssl/run.py ../my_dataset configs/vit_ssl_small_config.yaml all `
    --protocol protocol3 --label-budget 4shot --skip-sup --device cuda
~~~

`--skip-sup` 跳过 Sup-ViT 微调；当前命令仍会在每个划分进行 ViT-MAE 预训练和 MAE-ViT 微调。未指定 `--skip-sup` 时会同时运行 Sup-ViT。换用 P1/P2 时，同时修改审计命令和运行命令的 `--protocol`。

## 输出及结果汇总

- 基线实验：`result/`。
- MetaFi SSL 与审计：`result_metafi_ssl/runs/`。
- ViT-MAE：`result_metafi_ssl/vit/`。
- 汇总报告：`reports/RESULTS_SUMMARY.{md,csv,json}`；需要手动刷新时运行 `python scripts/report_results.py`。

以上目录均为运行后产生的本地文件，未提交到 GitHub。

## WiFi-JEPA 方法补充

本节在保留本 README 原有项目说明、目录介绍、环境要求和运行示例的基础上，补充本项目新增的 WiFi-JEPA 实现说明。

### 核心思路

WiFi-JEPA 原论文将 CSI 按子载波、时间和天线链路组织成三维结构，在时间-链路网格上遮挡完整链路，并让上下文编码器根据可见链路预测目标编码器产生的潜在特征。它不直接重建被遮挡的原始 CSI。目标编码器通过 EMA 更新，目标特征采用 LayerNorm 和停止梯度，预测器只在被遮挡位置计算 Smooth L1 损失。

原论文使用单天线发射器、三个接收器、每个接收器三根天线，共九条 Tx-Rx 链路，并输入幅度和去噪相位。论文将输入组织为 `(C,T,L)=(60,20,9)`，主实验遮挡五条链路、保留四条作为上下文。

### MM-Fi 适配

本项目的 MM-Fi CSI 发射端为单天线，接收端有三根天线。单帧幅度输入形状为：

```text
(3, 114, 10)
```

其中三路表示接收链路，114 为子载波数，10 为同一 CSI 窗口内的时间采样。模型将其重排为 `(C,T,L)=(114,10,3)`，生成 30 个时间-链路 token。

三路链路不能直接照搬论文的 5/9 遮挡比例。首版随机遮挡一整条链路、保留另外两条；被遮挡链路的全部 10 个时间 token 同时作为目标。双链路遮挡作为后续难度消融。

当前实现使用幅度输入并沿用项目现有预处理和 min-max 归一化。配置保留归一化消融开关。下游任务仍为 MM-Fi 单人 17 关节绝对坐标估计，复用项目的姿态回归头，不采用论文的多人物 PETR 解码器。

### 运行方式

WiFi-JEPA 代码位于 `WIFIJEPA/`。在仓库根目录运行：

预训练：

```powershell
python WIFIJEPA/scripts/pretrain.py --config WIFIJEPA/configs/matched_small.yaml
```

下游微调：

```powershell
python WIFIJEPA/scripts/finetune.py --config WIFIJEPA/configs/matched_small.yaml
```

配置项 `experiment.mode` 选择下游模式：`supervised` 使用随机初始化的结构化 ViT 作为纯监督对照；`jepa` 加载 WiFi-JEPA 预训练的 context encoder，再进行姿态监督微调。协议、划分、标签量、审计文件和结果路径由配置及自动路径逻辑共同确定。

## 数据集目录结构

导出仓库包含空的 `dataset/` 占位目录。将本地 MM-Fi 数据放入该目录，或在 WiFi-JEPA 配置中修改 `experiment.dataset_root`。预期结构为：

```text
dataset/
├─ E01/
│  └─ S01/
│     └─ A01/
│        ├─ ground_truth.npy
│        └─ wifi-csi/
│           ├─ frame001.mat
│           ├─ frame002.mat
│           └─ frame297.mat
├─ E02/
├─ E03/
└─ E04/
```

每个序列按场景、受试者和动作组织。CSI 单帧幅度形状为 `(3,114,10)`，姿态标签文件形状为 `(297,17,3)`。可选的 `wifi-csi-packed.npy` 可置于单个序列的 `wifi-csi/` 目录中以加速读取。

## 结果目录结构

仓库包含空的 `result/` 占位目录。数据审计、预训练 checkpoint、微调 checkpoint、指标和日志均为本地运行产物，不提交到 GitHub。

WiFi-JEPA 预训练结果示例：

```text
WIFIJEPA/results/pretrain/<protocol>/<split>/seed<seed>/<配置标识>/
├─ encoder.pth
├─ pretrain_manifest.json
└─ pretrain_metrics.yaml
```

WiFi-JEPA 下游结果示例：

```text
WIFIJEPA/result_metafi_ssl/<method>/<protocol>/<split>/seed<seed>/<标签预算>/<配置标识>/
├─ best_absolute.pth
├─ best_pelvis.pth
├─ best_pa.pth
├─ metrics.csv
├─ final_report.json
├─ test_outputs_absolute.npz
└─ done.txt
```

实际关节位置比较以 absolute checkpoint 和官方口径的 MPJPE 为主；骨盆对齐 MPJPE 与 PA-MPJPE 用于相对姿态诊断。

## WiFi-JEPA 后续训练计划

当前 WiFi-JEPA 结果暂不并入 ViT-MAE 结果表。后续先完成同结构、同数据边界、同标签量的配对实验：

1. 先在 Protocol 1、random split、seed42 下，将 20% strict 无标签序列的预训练由 10 epoch 补至 25 epoch。
2. 完成 2-shot 和 4-shot 下的 Sup-StructuredViT 与 WiFi-JEPA 对照。
3. 再扩展至 cross-subject 和 cross-scene 划分。
4. 若结果方向值得继续，再测试遮挡两条链路、多个随机种子及其他预训练目标。
5. 同时记录绝对 MPJPE、pelvis/PA 指标、骨盆位置偏差、姿态 spread、Corr51、各目标链路 loss 和优化步数。

只有当 WiFi-JEPA 在多个划分和标签量下相对同结构监督对照降低 absolute MPJPE，且不依赖单个随机种子时，才将收益描述为稳定改善。
