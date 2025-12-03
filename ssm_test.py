#双线
import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.metrics import HausdorffDistanceMetric
from skimage import measure
from ssm_dataloader import CardiacSegmentationDataset
from ssm_models import UNet3DWithSSM

# =============== 变量配置 ===============
root = "/home/guests/yilin_tang/Unet Fusion model v6/data/ED"
STRUCTURE_TYPE = "epi"   # "endo" or "epi"
STRUCTURES = ["baseline", "attn", "aux", "film"]
RESULT_ROOT = "/home/guests/yilin_tang/Unet Fusion model v6/results"

CHECKPOINTS = {
    "endo": {
        "baseline": "/home/guests/yilin_tang/Unet Fusion model v6/experiments/endo_baseline_20250615_004221/checkpoint_best.pth",
        "attn"    : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/endo_attn_20250615_010536/checkpoint_best.pth",
        "aux"     : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/endo_aux_20250615_012814/checkpoint_best.pth",
        "film"    : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/endo_film_20250615_015028/checkpoint_best.pth"
    },
    "epi": {
        "baseline": "/home/guests/yilin_tang/Unet Fusion model v6/experiments/epi_baseline_20250615_190550/checkpoint_best.pth",
        "attn"    : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/epi_attn_20250615_231746/checkpoint_best.pth",
        "aux"     : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/epi_aux_20250616_032701/checkpoint_best.pth",
        "film"    : "/home/guests/yilin_tang/Unet Fusion model v6/experiments/epi_film_20250616_073616/checkpoint_best.pth"
    }
}

N_VIS = 5   # 可视化前几个样本的三线图

def resolve_img_path(folder, scan_name):
    # 自动处理 .nii.gz/.nii 的重复和缺失
    base = scan_name
    if base.endswith('.nii.gz'):
        base = base[:-7]
    elif base.endswith('.nii'):
        base = base[:-4]
    p1 = os.path.join(folder, base + '.nii.gz')
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(folder, base + '.nii')
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Neither {p1} nor {p2} exists.")

def get_outer_contour_mask2d(binary_mask):
    contours = measure.find_contours(binary_mask, 0.5)
    contour_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for c in contours:
        c = np.round(c).astype(np.int32)
        for p in c:
            x, y = p
            x = np.clip(x, 0, binary_mask.shape[0] - 1)
            y = np.clip(y, 0, binary_mask.shape[1] - 1)
            contour_mask[int(x), int(y)] = 1
    return contour_mask

def get_outer_contour_mask3d(binary_mask3d):
    result = np.zeros_like(binary_mask3d, dtype=np.uint8)
    for z in range(binary_mask3d.shape[0]):
        result[z] = get_outer_contour_mask2d(binary_mask3d[z])
    return result

def visualize_overlay(img_vol, gt, pred, baseline, save_prefix, tag=""):
    slice_dict = {"x": img_vol.shape[0] // 2, "y": img_vol.shape[1] // 2, "z": img_vol.shape[2] // 2}
    for axis, idx in slice_dict.items():
        if axis == "x":
            img_slice  = img_vol[idx, :, :]
            gt_slice   = gt[idx, :, :]
            pred_slice = pred[idx, :, :]
            baseline_slice = baseline[idx, :, :]
        elif axis == "y":
            img_slice  = img_vol[:, idx, :]
            gt_slice   = gt[:, idx, :]
            pred_slice = pred[:, idx, :]
            baseline_slice = baseline[:, idx, :]
        elif axis == "z":
            img_slice  = img_vol[:, :, idx]
            gt_slice   = gt[:, :, idx]
            pred_slice = pred[:, :, idx]
            baseline_slice = baseline[:, :, idx]
        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice.T, cmap="gray", origin="lower")
        plt.contour(gt_slice.T, levels=[0.5], colors="white", linewidths=2)
        plt.contour(pred_slice.T, levels=[0.5], colors="blue", linewidths=2)
        plt.contour(baseline_slice.T, levels=[0.5], colors="red", linewidths=2)
        plt.title(f"{tag} Overlay {axis.upper()} - White: GT, Blue: Pred, Red: Baseline")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{axis}.png", dpi=200, bbox_inches='tight', pad_inches=0.05)
        plt.close()

def main():
    with open(os.path.join(root, "split_test.txt")) as f:
        test_ids = [line.strip() for line in f if line.strip()]
    latent_csv_path = os.path.join(root, f"latent_ssm_{STRUCTURE_TYPE}.csv")
    test_set = CardiacSegmentationDataset(os.path.join(root, "test"), latent_csv_path, test_ids)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssm_dim = test_set[0]['ssm'].shape[0]

    structure_map = {
        "baseline": {'use_film': False, 'use_aux': False, 'use_attention': False},
        "attn":     {'use_film': False, 'use_aux': False, 'use_attention': True},
        "aux":      {'use_film': False, 'use_aux': True,  'use_attention': False},
        "film":     {'use_film': True,  'use_aux': False, 'use_attention': False}
    }

    # baseline预测目录
    baseline_exp_dir = os.path.dirname(CHECKPOINTS[STRUCTURE_TYPE]["baseline"])
    baseline_exp_name = os.path.basename(baseline_exp_dir)
    baseline_pred_dir = os.path.join(RESULT_ROOT, f"{STRUCTURE_TYPE}_results", baseline_exp_name, "predictions")

    for structure in STRUCTURES:
        checkpoint_path = CHECKPOINTS[STRUCTURE_TYPE][structure]
        exp_dir = os.path.dirname(checkpoint_path)
        exp_name = os.path.basename(exp_dir)
        out_dir = os.path.join(RESULT_ROOT, f"{STRUCTURE_TYPE}_results", exp_name)
        os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
        log_fp = open(os.path.join(out_dir, "log.txt"), "w")

        # ===== 日志头 =====
        structure_desc = f"""
============================ SSM-Fusion {STRUCTURE_TYPE.capitalize()} Segmentation Experiment ============================
Structure                : {structure}
Experiment Name          : {exp_name}
Test cases               : {len(test_ids)}
Latent csv               : {latent_csv_path}
Latent SSM input shape   : {ssm_dim}
Predictions dir          : {os.path.join(out_dir, "predictions")}
===========================================================================================
{'id':>30} {'Dice':>8} {'Hausdorff':>10} {'Loss':>8}
"""
        log_fp.write(structure_desc)
        log_fp.flush()

        metrics = []
        model = UNet3DWithSSM(
            in_ch=1,
            base_ch=16,
            ssm_dim=ssm_dim,
            num_classes=3,
            **structure_map[structure]
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        hausdorff = HausdorffDistanceMetric(include_background=False, reduction="mean")
        for i, batch in enumerate(test_loader):
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            ssm = batch["ssm"].to(device)
            pred = model(img, ssm)
            pred_np = torch.argmax(pred, dim=1).cpu().numpy()[0]
            lbl_np = lbl.cpu().numpy()[0]
            img_np = img.cpu().numpy()[0,0]
            subj, scan = test_ids[i].split("/")

            # baseline mask
            baseline_mask_path = os.path.join(baseline_pred_dir, f"{i:03d}_{subj}_{scan}_{STRUCTURE_TYPE}_pred.nii.gz")
            if os.path.exists(baseline_mask_path):
                baseline_mask_raw = nib.load(baseline_mask_path).get_fdata().astype(np.uint8)
            else:
                baseline_mask_raw = np.zeros_like(pred_np)

            # epi/endo分割mask及后处理
            if STRUCTURE_TYPE == "epi":
                region_pred = ((pred_np == 1) | (pred_np == 2)).astype(np.uint8)
                region_gt   = ((lbl_np == 1) | (lbl_np == 2)).astype(np.uint8)
                mask_pred = get_outer_contour_mask3d(region_pred)
                mask_gt   = get_outer_contour_mask3d(region_gt)
                mask_baseline = get_outer_contour_mask3d(((baseline_mask_raw == 1) | (baseline_mask_raw == 2)).astype(np.uint8))
            elif STRUCTURE_TYPE == "endo":
                mask_pred = (pred_np == 2).astype(np.uint8)
                mask_gt   = (lbl_np == 2).astype(np.uint8)
                mask_baseline = (baseline_mask_raw == 2).astype(np.uint8)
            else:
                raise ValueError("Unknown STRUCTURE_TYPE!")

            # 使用健壮路径
            ref_img_path = resolve_img_path(os.path.join(root, "test", subj, "images"), scan)
            ref_nii = nib.load(ref_img_path)
            save_path = os.path.join(out_dir, "predictions", f"{i:03d}_{subj}_{scan}_{STRUCTURE_TYPE}_pred.nii.gz")
            nib.save(nib.Nifti1Image(mask_pred, ref_nii.affine), save_path)
            if i < N_VIS:
                plot_prefix = os.path.join(out_dir, "plots", f"{i:03d}_{subj}_{scan}_{STRUCTURE_TYPE}")
                visualize_overlay(img_np, mask_gt, mask_pred, mask_baseline, plot_prefix, tag=STRUCTURE_TYPE.upper())

            dice = 2 * np.sum(mask_pred * mask_gt) / (np.sum(mask_pred) + np.sum(mask_gt) + 1e-6)
            hausdorff.reset()
            haus_pred = torch.tensor(mask_pred[None,None], dtype=torch.float32)
            haus_gt   = torch.tensor(mask_gt[None,None], dtype=torch.float32)
            haus = hausdorff(haus_pred, haus_gt).item()
            out_mask = torch.softmax(pred, dim=1)[:,2] if STRUCTURE_TYPE == "endo" else torch.softmax(pred, dim=1)[:,1:].sum(1)
            gt_mask_tensor = torch.tensor(mask_gt[None], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy(out_mask, gt_mask_tensor).item()
            metrics.append({"id": test_ids[i], "dice": dice, "hausdorff": haus, "loss": loss})
            log_fp.write(f"{test_ids[i]:>30} {dice:8.4f} {haus:10.4f} {loss:8.4f}\n")
        pd.DataFrame(metrics).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        dice_mean = np.mean([m['dice'] for m in metrics])
        dice_std  = np.std([m['dice'] for m in metrics])
        haus_mean = np.mean([m['hausdorff'] for m in metrics])
        loss_mean = np.mean([m['loss'] for m in metrics])
        # ===== 日志结尾 summary =====
        summary = f"""
------------------ Experiment Summary [{exp_name}] --------------------
Structure           : {structure}
Total cases         : {len(test_ids)}
Best Dice           : {max([r['dice'] for r in metrics]):.4f}
Mean Dice           : {dice_mean:.4f}
Mean IoU            : Not calculated in this script
Mean Hausdorff      : {haus_mean:.4f}
Mean Loss           : {loss_mean:.4f}
Latent SSM input shape : {ssm_dim}
----------------------------------------------------------
"""
        log_fp.write(summary)
        log_fp.flush()
        log_fp.close()
        print(f"=== [{structure}] Done. Results in {out_dir}")

if __name__ == "__main__":
    main()
