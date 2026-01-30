import argparse
import numpy as np
import logging
import os
import sys
import multiprocessing
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

INPUT_ROOT = Path("/mnt/sdb/Bonnet-master/DS_ORIGINAL/")
OUTPUT_ROOT = Path("/mnt/sdb/Bonnet-master/data_pre/")
CONFIG_PATH = Path("/mnt/sdb/Bonnet-master/conf/classes.yaml")

HU_MIN = 200
HU_MAX = 3000
NUM_WORKERS = 8
TARGET_GROUP = 'all_classes'

def process_single_case(case_path: Path, output_root: Path, class_map: dict):
    case_name = case_path.name

    save_dir = output_root / case_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / "sparse_voxel.npz"

    if save_file.exists():
        return

    try:
        ct_path = case_path / "ct.nii.gz"
        if not ct_path.exists():
            return

        ct_nii = nib.load(str(ct_path))
        hu_voxels = ct_nii.get_fdata()

        labels = np.zeros(hu_voxels.shape, dtype=np.uint8)

        seg_dir = case_path / "segmentations"
        if not seg_dir.exists():
            seg_dir = case_path

        if seg_dir.exists():
            existing_files = set(os.listdir(seg_dir))

            for name, idx in class_map.items():
                if idx == 0: continue
                fname = f"{name}.nii.gz"

                if fname in existing_files:
                    try:
                        mask_nii = nib.load(str(seg_dir / fname))
                        mask_data = mask_nii.get_fdata()
                        labels[mask_data > 0.5] = idx
                    except Exception:
                        pass

        mask = (hu_voxels >= HU_MIN) & (hu_voxels <= HU_MAX)

        indices = np.argwhere(mask)

        voxel_features = hu_voxels[indices[:, 0], indices[:, 1], indices[:, 2]]
        voxel_labels = labels[indices[:, 0], indices[:, 1], indices[:, 2]]

        np.savez_compressed(
            save_file,
            voxel_features=voxel_features.astype(np.float32),
            voxel_coords=indices.astype(np.int32),
            voxel_labels=voxel_labels.astype(np.int32)
        )

    except Exception as e:
        print(f"Error processing {case_name}: {e}")

def main():
    if not CONFIG_PATH.exists():
        print(f"Error: Config not found at {CONFIG_PATH}")
        return

    conf = OmegaConf.load(CONFIG_PATH)
    if TARGET_GROUP in conf:
        class_map = OmegaConf.to_container(conf[TARGET_GROUP])
    else:
        class_map = OmegaConf.to_container(conf)

    print(f"=== Generate Cache Data ===")
    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Group: {TARGET_GROUP} ({len(class_map)} classes)")
    print(f"HU Range: {HU_MIN} ~ {HU_MAX}")
    print(f"Workers: {NUM_WORKERS}")

    all_cases = [
        INPUT_ROOT / d for d in os.listdir(INPUT_ROOT)
        if (INPUT_ROOT / d).is_dir() and d.startswith("s")
    ]
    all_cases.sort()
    print(f"Found {len(all_cases)} cases")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    process_func = partial(
        process_single_case,
        output_root=OUTPUT_ROOT,
        class_map=class_map
    )

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_func, all_cases, chunksize=1), total=len(all_cases)))

    print("✅ All Done!")

if __name__ == "__main__":
    main()