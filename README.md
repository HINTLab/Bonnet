# Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans

Bonnet is an ultra-fast whole-body bone segmentation pipeline for CT scans. It runs **in seconds per scan** on a single commodity GPU while maintaining reliable segmentation quality across different datasets.

---

## Contents

- [Training](#training)
- [Inference with sample data](#inference-with-sample-data)
- [Links: Processed training data & model weights](#links-processed-training-data--model-weights)

---

## Links: Processed training data & model weights

- **Processed training data** :  
  [Processed Data](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main/CACHE/batch_specs%3D(sparse_voxel)%2Chu_max%3D3000%2Chu_min%3D200)

- **Model weights** :  
  [Weights](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main)

---

## Training

1. Open the training config:

- `Bonnet/conf/config_eva.yaml`

2. Set your dataset / output paths and other options inside `config_eva.yaml` (e.g., point the dataset path to your local directory where you downloaded the **Processed Data** above).

3. Run training:

```bash
python main.py
```

## Inference with sample data

This repo includes **sample data** for inference. You only need to (1) point the data config to the correct local path, and (2) download weights and point the main config to the checkpoint.

### Step 1: Point the data config to the sample data

1. Open:

- `Bonnet/conf/data/totalseg_hu200_3000.yaml`

1. Update the dataset path fields in this file to the correct **local path** of the sample data in this GitHub repo.

> Tip: If the YAML contains multiple path keys (e.g., `data_root`, `dataset_root`, `img_dir`, etc.), update the one(s) used by your dataloader.

------

### Step 2: Download weights and set checkpoint path

1. Download the model checkpoint from Hugging Face:

- [Weights](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main)

1. Put the downloaded checkpoint under the `Bonnet/` directory (e.g., `Bonnet/checkpoints/`).
2. Open:

- `Bonnet/conf/config_eva.yaml`

1. Update the checkpoint field (e.g., `checkpoint_path` / `root_path` / `checkpoints_dir` depending on your config) to the correct **local path** of the downloaded weight file.

------

### Step 3: Enable eval-only mode and run

1. Open:

- `Bonnet/conf/eval/eval_on_test.yaml`

1. Set:

```
eval_only: True
```

1. Run:

```
python main.py
```
