# HPE: MM-Fi WiFi-CSI 3D Human Pose Estimation

This repository contains the core source code and configuration for MM-Fi WiFi-CSI 3D human pose estimation, including supervised baselines, existing SSL pipelines, ViT-MAE, and the WiFi-JEPA adaptation.

The repository does **not** include the MM-Fi dataset, model checkpoints, generated results, or training logs.

## Main entry points

Run commands from the repository root.

### Existing pipelines

Existing supervised/SSL entry points are under `scripts/`. Their usage and configuration options are documented in the corresponding YAML files and scripts.

### WiFi-JEPA

```powershell
python WIFIJEPA/scripts/pretrain.py --config WIFIJEPA/configs/matched_small.yaml
python WIFIJEPA/scripts/finetune.py --config WIFIJEPA/configs/matched_small.yaml
```

Set `experiment.mode` to `supervised` for the structured supervised control or to `jepa` for fine-tuning from the automatically selected WiFi-JEPA encoder checkpoint.

The default WiFi-JEPA configuration uses:

```text
protocol1 / strict / random_split / seed42 / 2shot
```

Change only the protocol, split, label budget, seed, or mode when switching experiments. Audit files and output directories are resolved automatically.

## WiFi-JEPA core implementation

```text
WIFIJEPA/
?? src/
?  ?? model.py       # Structured tokenizer, ViT, predictor, JEPA model, pose model
?  ?? masking.py     # Whole-link masking over the time-link grid
?  ?? data.py        # MM-Fi CSI loading and normalization ablation hook
?  ?? paths.py       # Automatic audit/checkpoint/result path resolution
?? scripts/
?  ?? pretrain.py    # WiFi-JEPA latent pre-training
?  ?? finetune.py    # Structured supervised or JEPA fine-tuning
?? configs/
?  ?? matched_small.yaml
?? tests/
   ?? test_core.py
```

The MM-Fi adaptation uses amplitude-only input with shape `(3, 114, 10)`, reorganized as `(C,T,L)=(114,10,3)`. It creates 30 time-link tokens and masks complete receiver links across all 10 time samples.

## Project structure

```text
HPE/
?? configs/          # Existing baseline and SSL configurations
?? dataset/          # Empty placeholder; put MM-Fi data here locally
?? mmfi_wifi/        # Data loading, training engine, metrics, manifests
?? pose_ssl/         # Existing SSL models and pre-training utilities
?? result/           # Empty placeholder for local generic results
?? scripts/          # Existing training, audit, and reporting entry points
?? WIFIJEPA/         # WiFi-JEPA core implementation and configuration
?? .gitignore
?? README.md
?? requirements.txt
```

Generated WiFi-JEPA artifacts are intentionally ignored by Git and remain local under:

```text
WIFIJEPA/audit/
WIFIJEPA/results/pretrain/
WIFIJEPA/result_metafi_ssl/
```

## Dataset structure

Set `experiment.dataset_root` in `WIFIJEPA/configs/matched_small.yaml` to the local MM-Fi dataset directory. The exported default is `dataset`.

```text
dataset/
?? E01/
?  ?? S01/
?     ?? A01/
?        ?? ground_truth.npy
?        ?? wifi-csi/
?           ?? frame001.mat
?           ?? frame002.mat
?           ?? frame297.mat
?? E02/
?? E03/
?? E04/
```

Each sequence is identified by `scene / subject / action`. CSI frames contain amplitude data with shape `(3, 114, 10)`. The ground-truth file contains 17 three-dimensional joints over 297 frames with shape `(297, 17, 3)`. An optional `wifi-csi-packed.npy` may be placed inside each `wifi-csi/` directory for packed frame loading.

## Result structure

The tracked `result/` directory is intentionally empty. Runtime outputs are ignored and remain local.

```text
result/
?? .gitkeep
```

WiFi-JEPA pre-training results:

```text
WIFIJEPA/results/pretrain/<protocol>/<split>/seed<seed>/<configuration-id>/
?? encoder.pth
?? pretrain_manifest.json
?? pretrain_metrics.yaml
```

WiFi-JEPA fine-tuning results:

```text
WIFIJEPA/result_metafi_ssl/<method>/<protocol>/<split>/seed<seed>/<label-budget>/<configuration-id>/
?? best_absolute.pth
?? best_pelvis.pth
?? best_pa.pth
?? metrics.csv
?? final_report.json
?? summary_absolute.json
?? summary_pelvis.json
?? summary_pa.json
?? test_outputs_absolute.npz
?? done.txt
```

The absolute checkpoint and `test_mpjpe_mm` are the primary outputs for actual joint position error. Pelvis-aligned MPJPE and PA-MPJPE are diagnostic metrics.

## Installation

Use Python 3.10 or newer. Install a PyTorch/torchvision pair compatible with the local CUDA driver, then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

The repository only contains code and configuration. Prepare the dataset locally before running training.
