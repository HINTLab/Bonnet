- # Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans

  Bonnet is an ultra-fast whole-body bone segmentation pipeline for CT scans. It runs **in seconds per scan** on a single commodity GPU while maintaining reliable segmentation quality across different datasets.**[[Paper\]](https://www.arxiv.org/abs/2601.22576)**

  ------

  ## Contents

  - [Processed training data & model weights](https://www.google.com/search?q=%23links-processed-training-data--model-weights)
  - [Data Preparation ](https://www.google.com/search?q=%23data-preparation)
  - [Training](https://www.google.com/search?q=%23training)
  - [Inference (Single Case vs. Sample Data)](https://www.google.com/search?q=%23inference)

  ------

  ## Processed training data & model weights

  - **Processed training data**: [Processed Data](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main/CACHE/batch_specs%3D(sparse_voxel)%2Chu_max%3D3000%2Chu_min%3D200)
  - **Model weights**: [Weights](https://huggingface.co/hanjiangjiang123/Bonnet/tree/main)

  ------

  ## Data Preparation

  If you want to train on your own dataset, use the provided preprocessing script to convert CT and segmentation files into sparse voxel formats (`.npz`).

  ### 1. Organize your raw data

  Ensure your data follows this structure:

  ```
  DS_ORIGINAL/
  ├── s0001/
  │   ├── ct.nii.gz
  │   └── segmentations/ 
  │       ├── vertebra_C1.nii.gz
  │       └── ...
  └── s0002/
  ```

  ### 2. Run Preprocessing

  The script `src/preprocess_dataset.py` extracts voxels within the [200, 3000] HU range and maps labels according to `conf/classes.yaml`.

  Bash

  ```
  python src/preprocess_dataset.py
  ```

  - **Input**: Defined by `INPUT_ROOT` in the script.
  - **Output**: Generates `sparse_voxel.npz` for each case in the `OUTPUT_ROOT`.

  ------

  ## Training

  1. **Configure Paths**:
     - Open `Bonnet/conf/data/totalseg_hu200_3000.yaml`.
     - Set `dataset_path` and `cache_path` to your **preprocessed data** folder.
  2. **Run Training**:

  ```
  python main.py
  ```

  ------

  ## Inference

  ### Option A: Single Case Inference (End-to-End)

  Use `inference_single.py` to segment a raw `.nii.gz` CT scan directly. This script handles preprocessing, windowed inference, and volume reconstruction automatically.

  Bash

  ```
  python inference_single.py --ct /path/to/ct.nii.gz --out ./prediction.nii.gz --device cuda
  ```

  > **Note**: Ensure the `CHECKPOINT_PATH` and `CONFIG_PATH` inside the script point to your actual local files.

  ### Option B: Evaluation with Sample Data

  To run evaluation on the provided sample test set:

  1. Open `Bonnet/conf/eval/eval_on_test.yaml` and set `eval_only: True`.
  2. Point your data config to the sample data path.
  3. Run: `python main.py`

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhu2026bonnet,
  title={Bonnet: Ultra-Fast Whole-Body Bone Segmentation from CT Scans},
  author={Hanjiang Zhu and Pedro Martelleto Rezende and Zhang Yang and Tong Ye and Bruce Z. Gao and Feng Luo and Siyu Huang and Jiancheng Yang},
  journal={arXiv preprint arXiv:2601.22576},
  year={2026}
}
