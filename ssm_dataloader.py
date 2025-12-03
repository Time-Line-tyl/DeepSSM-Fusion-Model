import sys
sys.path.insert(0, '/home/guests/yilin_tang/Unet Fusion model v5/ShapeWorks-v6.4.2-linux/Python')
sys.path.insert(0, '/home/guests/yilin_tang/Unet Fusion model v5/ShapeWorks-v6.4.2-linux/bin')

import os
import shutil
import numpy as np
# 修补 numpy.bool 弃用问题，确保 ITK 模块能正常导入
np.bool = bool
import pandas as pd
from pathlib import Path
import nibabel as nib
import torch
from DeepSSMUtils import model
from scipy.ndimage import zoom
import shapeworks as sw
from torch.utils.data import Dataset
import numpy as np

def extract_subject_id(filename):
    return '_'.join(filename.split('_')[:2])  # e.g., "MITEA_001"

def move_to_target(src_images, src_labels, dst_root, tag):
    images_dir = Path(dst_root) / "images"
    labels_dir = Path(dst_root) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in zip(src_images, src_labels):
        if img_path.name.endswith(f"_{tag}.nii.gz"):
            shutil.copy(img_path, images_dir / img_path.name)
            shutil.copy(lbl_path, labels_dir / lbl_path.name)

def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images_dir = Path(root_dir) / "images"
    images = sorted(list(images_dir.glob("*.nii.gz")))
    subjects = sorted({extract_subject_id(f.name) for f in images})
    np.random.seed(42)
    np.random.shuffle(subjects)
    n_total = len(subjects)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_subj = subjects[:n_train]
    val_subj = subjects[n_train:n_train + n_val]
    test_subj = subjects[n_train + n_val:]
    split_map = {"train": train_subj, "val": val_subj, "test": test_subj}
    for split, subj_list in split_map.items():
        for subj in subj_list:
            subj_img_dir = Path(root_dir) / split / subj / "images"
            subj_lbl_dir = Path(root_dir) / split / subj / "labels"
            subj_img_dir.mkdir(parents=True, exist_ok=True)
            subj_lbl_dir.mkdir(parents=True, exist_ok=True)
            for img in images_dir.glob(f"{subj}_*.nii.gz"):
                lbl = Path(root_dir) / "labels" / img.name
                shutil.copy(img, subj_img_dir / img.name)
                shutil.copy(lbl, subj_lbl_dir / lbl.name)
    # 生成 split_*.txt
    for split in split_map:
        split_dir = Path(root_dir) / split
        with open(Path(root_dir) / f"split_{split}.txt", 'w') as f:
            for subj in split_dir.iterdir():
                if subj.is_dir():
                    for nii in (subj / "images").glob("*.nii.gz"):
                        if nii.name.startswith("._"):
                            continue
                        relative_id = f"{subj.name}/{nii.stem}"
                        f.write(f"{relative_id}\n")

def load_encoder(config_dir, encoder_type):
    config_path = os.path.join(config_dir, "mitea_deepssm.json")
    model_path = os.path.join(config_dir, "mitea_deepssm", "best_model.torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if encoder_type == "tl":
        net_full = model.DeepSSMNet_TLNet(config_path)
        net_full.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net = net_full.ImageEncoder
    elif encoder_type == "base":
        net_full = model.DeepSSMNet(config_path)
        net_full.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        net = net_full.encoder
    else:
        raise ValueError("Unknown encoder_type")
    net.eval()
    net.to(device)
    for p in net.parameters():
        p.requires_grad = False
    return net, device

def extract_latent(encoder, img_path, device):
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    # ----------- 自动获取 target shape 并 resize -----------
    if hasattr(encoder, "img_dims"):
        # 有的模型保存为 list, tuple, numpy array等，都能兼容
        try:
            target_shape = tuple(int(x) for x in encoder.img_dims)
        except Exception:
            target_shape = (112, 128, 128)
    else:
        target_shape = (112, 128, 128)
    if img.shape != target_shape:
        factors = [t / s for t, s in zip(target_shape, img.shape)]
        img = zoom(img, zoom=factors, order=1)
    # ----------- 转张量并转 device -----------
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        zt, _ = encoder(img)
    return zt.cpu().numpy().flatten()


def extract_latent_vectors(root_dir, which="epi"):
    if which == "epi":
        tl_encoder, device_tl = load_encoder(
            "/home/guests/yilin_tang/Unet Fusion model v5/Output/deep_ssm_mitea/tl_epi", "tl"
        )
        base_encoder, device_base = load_encoder(
            "/home/guests/yilin_tang/Unet Fusion model v5/Output/deep_ssm_mitea/base_epi", "base"
        )
    elif which == "endo":
        tl_encoder, device_tl = load_encoder(
            "/home/guests/yilin_tang/Unet Fusion model v5/Output/deep_ssm_mitea/tl_endo", "tl"
        )
        base_encoder, device_base = load_encoder(
            "/home/guests/yilin_tang/Unet Fusion model v5/Output/deep_ssm_mitea/base_endo", "base"
        )
    else:
        raise ValueError("which must be epi or endo")
    print(f"🔍 Extracting {which} latent vectors (tl_{which} + base_{which}) ...")
    rows = []
    for split in ["train", "val", "test"]:
        for subj_dir in (Path(root_dir) / split).iterdir():
            if not subj_dir.is_dir():
                continue
            for img_path in (subj_dir / "images").glob("*.nii*"):
                subj_id = subj_dir.name
                scan_name = img_path.stem  # 不带后缀
                full_id = f"{subj_id}/{scan_name}"  # 完全和 split_xxx.txt 统一
                z_tl = extract_latent(tl_encoder, img_path, device_tl)
                z_base = extract_latent(base_encoder, img_path, device_base)
                rows.append([full_id] + z_tl.tolist() + z_base.tolist())
    latent_dim = len(z_tl)
    columns = ["id"] + [f"tl_{which}_{i}" for i in range(latent_dim)] + [f"base_{which}_{i}" for i in range(latent_dim)]
    df = pd.DataFrame(rows, columns=columns)
    df.set_index("id", inplace=True)
    df.to_csv(Path(root_dir) / f"latent_ssm_{which}.csv")
    print(f"✅ Saved latent vectors to {Path(root_dir) / f'latent_ssm_{which}.csv'}")


if __name__ == "__main__":
    src_root = "/home/guests/yilin_tang/Unet Fusion model v6/data/MITEA"
    src_images = sorted(list((Path(src_root) / "images").glob("*.nii.gz")))
    src_labels = sorted([Path(src_root) / "labels" / f.name for f in src_images])

    # ----- ED -----
    ed_root = "/home/guests/yilin_tang/Unet Fusion model v6/data/ED"
    move_to_target(src_images, src_labels, ed_root, "ED")
    split_dataset(ed_root)
    extract_latent_vectors(ed_root, "epi")
    extract_latent_vectors(ed_root, "endo")

    # ----- ES -----
    es_root = "/home/guests/yilin_tang/Unet Fusion model v6/data/ES"
    move_to_target(src_images, src_labels, es_root, "ES")
    split_dataset(es_root)
    extract_latent_vectors(es_root, "epi")
    extract_latent_vectors(es_root, "endo")


# ------- Dataset ---------
class CardiacSegmentationDataset(Dataset):
    """
    3D医学分割数据集，输出包括
      - 图像: [1, D, H, W]
      - 标签: [D, H, W]
      - SSM向量: [N]（csv的有效列数）
    """
    def __init__(self, split_dir, latent_csv, id_list, target_shape=(112, 128, 128)):
        self.split_dir = split_dir
        self.ids = id_list
        self.target_shape = target_shape
        self.latent_df = pd.read_csv(latent_csv, index_col=0)
        # 保证读取所有列
        print(f"[INFO] Loaded latent SSM shape: {self.latent_df.shape}")  # 如(268,28)

    def _resize(self, v):
        factors = [t / s for t, s in zip(self.target_shape, v.shape)]
        return zoom(v, zoom=factors, order=1)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        subj, scan_name = _id.split('/')
        if scan_name.endswith(".nii.gz"):
            raw = scan_name[:-7]
        elif scan_name.endswith(".nii"):
            raw = scan_name[:-4]
        else:
            raw = scan_name
        img_path = os.path.join(self.split_dir, subj, "images", raw + ".nii.gz")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.split_dir, subj, "images", raw + ".nii")
        img = nib.load(img_path).get_fdata().astype(np.float32)
        img = np.expand_dims(self._resize(img), 0)  # [1, D, H, W]
        lbl_path = os.path.join(self.split_dir, subj, "labels", raw + ".nii.gz")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(self.split_dir, subj, "labels", raw + ".nii")
        lbl = nib.load(lbl_path).get_fdata().astype(np.int32)
        lbl = self._resize(lbl).round().astype(np.int32)  # [D, H, W]
        ssm = self.latent_df.loc[_id].values.astype(np.float32)
        return {
            "image": torch.from_numpy(img),        # [1, D, H, W]
            "label": torch.from_numpy(lbl).long(), # [D, H, W]
            "ssm": torch.from_numpy(ssm)           # [latent_dim]
        }

    def __len__(self):
        return len(self.ids)


