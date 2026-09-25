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

以上目录均为运行后产生的本地文件，已被 `.gitignore` 排除，不应提交到 GitHub。未附权重时需重新预训练，不能直接跳过预训练运行微调。
