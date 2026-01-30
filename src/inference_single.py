import torch
import numpy as np
import nibabel as nib
import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from models.sparse_unet import SparseUNet
from data import test_patches_provider
from data import data_utils
from models.models_utils import gaussian_kernel

GENERAL_MEAN = -104.43730926513672
GENERAL_STD = 505.3545227050781

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = "/mnt/sdb/Bonnet-master/conf/classes.yaml"
CHECKPOINT_PATH = "/mnt/sdb/Bonnet-master/spconv_unet_04.11.2025Y-21:28:25_8219179675798613560/best_model.pth"
DEFAULT_INPUT = "/mnt/sdb/Bonnet-master/DS_ORIGINAL/s0001/ct.nii.gz"
DEFAULT_OUTPUT = "/mnt/sdb/Bonnet-master/prediction_s0001.nii.gz"


def run_inference(ct_path, output_path, device_str='cuda'):
    device = torch.device(device_str)

    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Config not found: {CONFIG_PATH}")
        return

    try:
        full_conf = OmegaConf.load(CONFIG_PATH)

        target_group = 'all_classes'

        if target_group in full_conf:
            class_map = OmegaConf.to_container(full_conf[target_group])
        else:
            class_map = OmegaConf.to_container(full_conf)

        num_classes = len(class_map)
        logging.info(f"✅ Loaded {num_classes} classes ({target_group}) to match checkpoint.")
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return

    logging.info(f"Loading CT: {ct_path}")
    nimg = nib.load(ct_path)
    dense_vol = nimg.get_fdata()

    logging.info("Preprocessing (voxel_to_cloud)...")
    dummy_input_labels = np.zeros_like(dense_vol, dtype=np.int32)

    voxel_coords, _, voxel_feats = data_utils.voxel_to_cloud(
        dense_vol, dummy_input_labels, hu_min=200, hu_max=3000
    )

    if voxel_coords.shape[0] == 0:
        logging.error("No valid voxels found! Check HU range.")
        return

    logging.info("Normalizing features...")
    voxel_coords = torch.from_numpy(voxel_coords).int()
    voxel_features = torch.from_numpy(voxel_feats).float()

    voxel_features = (voxel_features - GENERAL_MEAN) / GENERAL_STD
    voxel_features = voxel_features.unsqueeze(-1)  # [N, 1]

    dummy_labels = torch.zeros(voxel_coords.shape[0], dtype=torch.int32)

    logging.info("Generating patches...")
    WINDOW_SIZE = [128, 128, 128]
    OVERLAP = 0.5
    win_size_tensor = torch.tensor(WINDOW_SIZE)

    patches = test_patches_provider.get_all_patches(
        voxel_coords,
        voxel_features,
        dummy_labels,
        window_size=win_size_tensor,
        test_window_overlap=OVERLAP
    )
    logging.info(f"Generated {len(patches)} patches.")

    logging.info("Loading model...")

    model = SparseUNet(num_classes=num_classes, window_size=WINDOW_SIZE, width=4.0).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        logging.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get('runner_state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace('segmentator.', '')
        new_state_dict[key] = v

    try:
        model.load_state_dict(new_state_dict)
        logging.info("✅ Model weights loaded successfully!")
    except RuntimeError as e:
        logging.error(f"❌ Weight mismatch! Please fix models/sparse_unet.py.")
        raise e

    model.eval()

    logging.info("Running inference...")

    num_points = voxel_coords.shape[0]
    full_logits = torch.zeros((num_points, num_classes), device=device)
    count_map = torch.zeros((num_points), device=device)

    window_size_cuda = win_size_tensor.to(device).float()

    with torch.no_grad():
        for patch in tqdm(patches, desc="Inference"):
            p_coords = patch['voxel_coords'].to(device)
            p_feats = patch['voxel_features'].to(device)
            p_indices = patch['window_indices'].to(device)

            batch_idx = torch.zeros((p_coords.shape[0], 1), device=device, dtype=torch.int32)
            p_coords_batch = torch.cat([batch_idx, p_coords], dim=1)

            input_dict = {
                'voxel_coords': p_coords_batch,
                'voxel_features': p_feats,
                'batch_size': 1
            }

            logits = model(input_dict)

            weight = gaussian_kernel(p_coords, window_size_cuda, std_scale=0.5)
            logits = logits * weight.unsqueeze(1)

            idx_expand = p_indices.unsqueeze(1).expand(-1, num_classes).long()
            full_logits.scatter_add_(0, idx_expand, logits)
            count_map.scatter_add_(0, p_indices.long(), weight)

    logging.info("Reconstructing volume...")

    count_map[count_map == 0] = 1.0
    full_logits /= count_map.unsqueeze(1)

    pred_labels_sparse = torch.argmax(full_logits, dim=1).cpu().numpy().astype(np.uint8)

    indices = voxel_coords.numpy()

    pred_vol = np.zeros(dense_vol.shape, dtype=np.uint8)
    pred_vol[indices[:, 0], indices[:, 1], indices[:, 2]] = pred_labels_sparse

    save_nii = nib.Nifti1Image(pred_vol, nimg.affine)

    save_nii.header.set_qform(nimg.affine, code=1)
    save_nii.header.set_sform(nimg.affine, code=1)

    nib.save(save_nii, output_path)

    logging.info(f"✅ Prediction saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_inference(args.ct, args.out, args.device)