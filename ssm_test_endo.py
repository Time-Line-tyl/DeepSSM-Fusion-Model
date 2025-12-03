import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from ssm_dataloader import CardiacSegmentationDataset
from ssm_models import UNet3DWithSSM

# =========== 配置 ===========
root = "/home/guests/yilin_tang/Unet Fusion model v6/data/ED"
latent_csv = os.path.join(root, "latent_ssm_endo.csv")
split_txt = os.path.join(root, "split_test.txt")

# ======= 明确实验和权重参数 =======
exp_name = "endo_attn_20250614_195337"
exp_baseline = "endo_baseline_20250614_193022"

ckpt_baseline = f"/home/guests/yilin_tang/Unet Fusion model v6/experiments/{exp_baseline}/checkpoint_best.pth"
ckpt_fusion   = f"/home/guests/yilin_tang/Unet Fusion model v6/experiments/{exp_name}/checkpoint_best.pth"
cfg_baseline  = dict(use_film=False, use_aux=False, use_attention=False)
cfg_fusion    = dict(use_film=False, use_aux=False, use_attention=True)   # 这里是attn，对aux/film只要对应调整

# ======= 结果文件夹自动归类 =======
save_root = f"/home/guests/yilin_tang/Unet Fusion model v6/endo_results/{exp_name}"
plots_dir = os.path.join(save_root, "plots")            # overlay三线论文图
pred_dir_baseline = os.path.join(save_root, "predictions_baseline")
pred_dir_fusion   = os.path.join(save_root, "predictions_fusion")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(pred_dir_baseline, exist_ok=True)
os.makedirs(pred_dir_fusion, exist_ok=True)

# ======= 加载数据 =======
with open(split_txt) as f:
    test_ids = [line.strip() for line in f if line.strip()]
test_set = CardiacSegmentationDataset(os.path.join(root, "test"), latent_csv, test_ids)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= 加载模型 =======
model_baseline = UNet3DWithSSM(
    in_ch=1, base_ch=16, ssm_dim=test_set.latent_df.shape[1], num_classes=3, **cfg_baseline
).to(device)
model_baseline.load_state_dict(torch.load(ckpt_baseline, map_location=device))
model_baseline.eval()

model_fusion = UNet3DWithSSM(
    in_ch=1, base_ch=16, ssm_dim=test_set.latent_df.shape[1], num_classes=3, **cfg_fusion
).to(device)
model_fusion.load_state_dict(torch.load(ckpt_fusion, map_location=device))
model_fusion.eval()

# ======= 工具函数 =======
def get_epi_mask(pred):
    epi_mask = pred.copy()
    epi_mask[epi_mask == 2] = 1
    return (epi_mask > 0).astype(np.uint8)

def plot_compare_epi_3views(image, gt, baseline_pred, fusion_pred, save_path, case_title=""):
    gt_epi = get_epi_mask(gt)
    baseline_epi = get_epi_mask(baseline_pred)
    fusion_epi = get_epi_mask(fusion_pred)
    views = ["axial", "coronal", "sagittal"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, v in enumerate(views):
        if v == "axial":
             idx = image.shape[0]//2
             img_sl = image[idx]
             gt_sl = gt_epi[idx]
             baseline_sl = baseline_epi[idx]
             fusion_sl = fusion_epi[idx]
        elif v == "coronal":
             idx = image.shape[1]//2
             img_sl = image[:, idx, :]
             gt_sl = gt_epi[:, idx, :]
             baseline_sl = baseline_epi[:, idx, :]
             fusion_sl = fusion_epi[:, idx, :]
             # 逆时针旋转90度
             img_sl = np.rot90(img_sl, k=1)
             gt_sl = np.rot90(gt_sl, k=1)
             baseline_sl = np.rot90(baseline_sl, k=1)
             fusion_sl = np.rot90(fusion_sl, k=1)
        else:  # sagittal
             idx = image.shape[2]//2
             img_sl = image[:, :, idx]
             gt_sl = gt_epi[:, :, idx]
             baseline_sl = baseline_epi[:, :, idx]
             fusion_sl = fusion_epi[:, :, idx]
             img_sl = np.rot90(img_sl, k=1)
             gt_sl = np.rot90(gt_sl, k=1)
             baseline_sl = np.rot90(baseline_sl, k=1)
             fusion_sl = np.rot90(fusion_sl, k=1)
        axs[i].imshow(img_sl, cmap="gray")
        axs[i].contour(gt_sl, levels=[0.5], colors="white", linewidths=2)
        axs[i].contour(baseline_sl, levels=[0.5], colors="blue", linewidths=2)
        axs[i].contour(fusion_sl, levels=[0.5], colors="red", linewidths=2)
        axs[i].set_title(v.capitalize())
        axs[i].axis("off")

    handles = [
        plt.Line2D([0], [0], color='white', lw=2, label='GT'),
        plt.Line2D([0], [0], color='blue',  lw=2, label='Baseline'),
        plt.Line2D([0], [0], color='red',   lw=2, label='Fusion')
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    plt.suptitle(case_title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def dice_coef(pred, target, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_coef(pred, target, eps=1e-6):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    return (inter + eps) / (union + eps)

# ======= 推理/保存 =======
results = []
start_time = datetime.now()
with torch.no_grad():
    for i in range(len(test_set)):
        batch = test_set[i]
        img  = batch["image"].unsqueeze(0).to(device)
        ssm  = batch["ssm"].unsqueeze(0).to(device)
        label= batch["label"].cpu().numpy()
        img_np = img.cpu().numpy()[0, 0]

        out_baseline = model_baseline(img, ssm)
        pred_baseline = torch.argmax(out_baseline, dim=1).cpu().numpy()[0]
        out_fusion = model_fusion(img, ssm)
        pred_fusion = torch.argmax(out_fusion, dim=1).cpu().numpy()[0]

        case_id = test_ids[i].replace("/", "_")
        # === 论文三线overlay ===
        plot_path = os.path.join(plots_dir, f"{i:02d}_{case_id}_epi_compare_3views.png")
        plot_compare_epi_3views(img_np, label, pred_baseline, pred_fusion, plot_path, case_title=case_id)
        # === 保存nii ===
        affine = np.eye(4)
        try:
            subj, scan = test_ids[i].split("/")
            label_path = os.path.join(root, "test", subj, "labels", scan + ".nii.gz")
            if os.path.exists(label_path):
                affine = nib.load(label_path).affine
        except: pass
        nii_path_baseline = os.path.join(pred_dir_baseline, f"{case_id}_unet.nii.gz")
        nii_path_fusion   = os.path.join(pred_dir_fusion, f"{case_id}_fusion.nii.gz")
        nib.save(nib.Nifti1Image(pred_baseline.astype(np.uint8), affine), nii_path_baseline)
        nib.save(nib.Nifti1Image(pred_fusion.astype(np.uint8), affine), nii_path_fusion)
        # === 指标
        gt_epi = get_epi_mask(label)
        baseline_epi = get_epi_mask(pred_baseline)
        fusion_epi = get_epi_mask(pred_fusion)
        dice_baseline = dice_coef(baseline_epi, gt_epi)
        dice_fusion = dice_coef(fusion_epi, gt_epi)
        iou_baseline = iou_coef(baseline_epi, gt_epi)
        iou_fusion = iou_coef(fusion_epi, gt_epi)
        results.append({
            "case": case_id,
            "dice_baseline": dice_baseline,
            "dice_fusion": dice_fusion,
            "iou_baseline": iou_baseline,
            "iou_fusion": iou_fusion,
            "overlay": plot_path,
            "nii_baseline": nii_path_baseline,
            "nii_fusion": nii_path_fusion
        })
        print(f"[{i+1}/{len(test_set)}] {case_id}: Dice Baseline={dice_baseline:.4f}, Fusion={dice_fusion:.4f}")

# 保存csv和log
metrics_df = pd.DataFrame(results)
csv_path = os.path.join(save_root, "metrics.csv")
metrics_df.to_csv(csv_path, index=False)
summary = (
    f"Test Baseline: {ckpt_baseline}\n"
    f"Test Fusion:   {ckpt_fusion}\n"
    f"Samples: {len(test_set)}\n"
    f"Mean Dice Baseline: {metrics_df['dice_baseline'].mean():.4f} ± {metrics_df['dice_baseline'].std():.4f}\n"
    f"Mean Dice Fusion:   {metrics_df['dice_fusion'].mean():.4f} ± {metrics_df['dice_fusion'].std():.4f}\n"
    f"Mean IoU Baseline:  {metrics_df['iou_baseline'].mean():.4f} ± {metrics_df['iou_baseline'].std():.4f}\n"
    f"Mean IoU Fusion:    {metrics_df['iou_fusion'].mean():.4f} ± {metrics_df['iou_fusion'].std():.4f}\n"
    f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
)
with open(os.path.join(save_root, "log.txt"), "w") as f:
    f.write(summary)
    f.write(metrics_df.to_string(index=False))
print(summary)
print("已保存:", plot_path)
print("已保存:", nii_path_baseline, nii_path_fusion)
print("✅ 测试结束！")
