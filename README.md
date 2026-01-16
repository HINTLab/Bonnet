# Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans

Bonnet is an ultra-fast whole-body bone segmentation pipeline for CT scans. It runs **in seconds per scan** on a single commodity GPU while maintaining reliable segmentation quality across different datasets.

---

## Contents

- [Links: Processed training data & model weights](#links-processed-training-data--model-weights)
- [Training](#training)
- [Inference with sample data](#inference-with-sample-data)

---

## Links: Processed training data & model weights

- **Processed training data**:  
  [Processed Data](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main/CACHE/batch_specs%3D(sparse_voxel)%2Chu_max%3D3000%2Chu_min%3D200)

- **Model weights**:  
  [Weights](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main)

---

## Training

1. Open the configs:

   - `Bonnet/conf/config_eva.yaml`
   - `Bonnet/conf/data/totalseg_hu200_3000.yaml`

2. Set paths and options:

   - In `Bonnet/conf/config_eva.yaml`, configure your output/log paths and other training options. Make sure the dataset selection points to the correct data config you are using.
   - In `Bonnet/conf/data/totalseg_hu200_3000.yaml`, set the local paths for `dataset_path` and `cache_path` to match your machine.

3. Run training:

```bash
python main.py
```

## Inference with sample data

This repo includes **sample data** for inference. You only need to:

1. point the data config to the correct local path.
2. download weights and point the main config to the checkpoint.

### Step 1: Point the data config to the sample data

1. Open:
   - `Bonnet/conf/data/totalseg_hu200_3000.yaml`
2. Set `dataset_path` and `cache_path` to the correct **local path** of the sample data in this GitHub repo.

### Step 2: Download weights and set checkpoint path

1. Download the model checkpoint from:
   - [Weights](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main)
2. Put the downloaded checkpoint under the `Bonnet/` directory.
3. Open:
   - `Bonnet/conf/config_eva.yaml`
4. Update the checkpoint field in this config (e.g., `checkpoint_path` / `root_path` / `checkpoints_dir`, depending on your config) to the correct **local path** of the downloaded weight file.

### Step 3: Enable eval-only mode and run

1. Open:
   - `Bonnet/conf/eval/eval_on_test.yaml`
2. Set:
```
eval_only: True
```
3. Run:
```
python main.py
```
