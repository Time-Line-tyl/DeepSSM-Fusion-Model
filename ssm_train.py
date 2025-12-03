import os, sys, copy, time, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ssm_dataloader import CardiacSegmentationDataset
from ssm_models import UNet3DWithSSM
from monai.metrics import HausdorffDistanceMetric



# ========== 参数设置 ==========
root = "/home/guests/yilin_tang/Unet Fusion model v6/data/ED"
latent_csv = os.path.join(root, "latent_ssm_endo.csv")
exp_type = "endo"  # "endo" or "epi"
STRUCTURES = ["baseline", "attn", "aux", "film"] #
num_epochs = 100
batch_size = 2
learning_rate = 1e-4


# ---- 数据采样相关参数 ----
NUM_TRAIN = 64   # 训练集图片数
NUM_VAL   = 32    # 验证集图片数
NUM_TEST  = 32    # 测试集图片数
RANDOM_SEED = 42  # 随机种子

# ---- 训练日志/模型保存相关 ----
save_every_epoch = 50          # 每多少epoch存一次checkpoint
update_best_every_epoch = 20     # 每多少epoch检查一次是否更新best model
log_iter_interval = 16           # 每多少iter输出一次log
# ==================================

def timestamp_str():
    return datetime.now().strftime("[%H:%M:%S]")

def load_ids(split):
    txt = os.path.join(root, f"split_{split}.txt")
    with open(txt) as f:
        return [line.strip() for line in f if line.strip()]

# 随机采样数据
train_ids = load_ids("train")
val_ids   = load_ids("val")
test_ids  = load_ids("test")
random.seed(RANDOM_SEED)
if NUM_TRAIN < len(train_ids):
    train_ids = random.sample(train_ids, NUM_TRAIN)
if NUM_VAL < len(val_ids):
    val_ids = random.sample(val_ids, NUM_VAL)
if NUM_TEST < len(test_ids):
    test_ids = random.sample(test_ids, NUM_TEST)

num_train_imgs = len(train_ids)
num_val_imgs = len(val_ids)
num_test_imgs = len(test_ids)

train_set = CardiacSegmentationDataset(os.path.join(root, "train"), latent_csv, train_ids)
val_set   = CardiacSegmentationDataset(os.path.join(root, "val"),   latent_csv, val_ids)
test_set  = CardiacSegmentationDataset(os.path.join(root, "test"),  latent_csv, test_ids)
loaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0),  # num_workers=0保证兼容所有环境
    'val':   DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0),
    'test':  DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=0)
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

structure_map = {
    "baseline": {'use_film': False, 'use_aux': False, 'use_attention': False},
    "attn":     {'use_film': False, 'use_aux': False, 'use_attention': True},
    "aux":      {'use_film': False, 'use_aux': True,  'use_attention': False},
    "film":     {'use_film': True,  'use_aux': False, 'use_attention': False}
}

def dice_coef(pred, target, num_classes=3, eps=1e-6):
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * inter + eps) / (union + eps)
        dices.append(dice.item())
    return dices

def iou_coef(pred, target, eps=1e-6):
    inter = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return inter / (union + eps)

for structure in STRUCTURES:
    cfg = structure_map[structure]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp_type}_{structure}_{timestamp}"
    exp_dir = os.path.join("/home/guests/yilin_tang/Unet Fusion model v6/experiments", exp_name)
    log_dir = os.path.join(exp_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train_log.txt")
    log_fp = open(log_file, "w")

    class Logger(object):
        def __init__(self, *files): self.files = files
        def write(self, message): [f.write(message) for f in self.files if f]; [f.flush() for f in self.files if f]
        def flush(self): [f.flush() for f in self.files if f]
    sys.stdout = Logger(sys.stdout, log_fp)  # 双写日志

    # ===== 参数描述输出（只输出一次） =====
    structure_desc = f"""
============================ SSM-Fusion Segmentation Experiment ============================
Structure                : {structure}
Time                     : {timestamp}
Epochs                   : {num_epochs}
Batch size               : {batch_size}
Iterations per epoch     : {len(loaders['train'])}
Train images             : {num_train_imgs}
Val images               : {num_val_imgs}
Test images              : {num_test_imgs}
Learning rate (Adam)     : {learning_rate}
Loss function            : CrossEntropyLoss
Latent SSM input shape   : {train_set.latent_df.shape[1]}
Random Seed              : {RANDOM_SEED}
Network config           : {cfg}
save_every_epoch         : {save_every_epoch}
update_best_every_epoch  : {update_best_every_epoch}
log_iter_interval        : {log_iter_interval}
===========================================================================================
"""
    print(structure_desc)
    log_fp.write(structure_desc)
    log_fp.flush()

    model_net = UNet3DWithSSM(
        in_ch=1,
        base_ch=16,
        ssm_dim=train_set.latent_df.shape[1],
        num_classes=3,
        **cfg
    ).to(device)
    optimizer = optim.Adam(model_net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer    = SummaryWriter(log_dir=log_dir)
    best_w, best_dice = copy.deepcopy(model_net.state_dict()), 0.0

    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_hausdorff': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_hausdorff': []
    }

    for epoch in range(num_epochs):
        model_net.train()
        running_loss = running_dice = running_iou = running_hausdorff = n = 0
        iter_count = 0
        for batch in loaders['train']:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            ssm = batch['ssm'].to(device)
            optimizer.zero_grad()
            out = model_net(img, ssm)
            loss = criterion(out, lbl)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * img.size(0)
            pred = torch.argmax(out, dim=1)
            dice = np.mean(dice_coef(pred, lbl))
            iou = iou_coef((pred > 0), (lbl > 0))
            hausdorff_metric.reset()
            hausdorff_metric(pred, lbl)
            hausdorff = hausdorff_metric.aggregate().item()
            running_dice += dice * img.size(0)
            running_iou += iou * img.size(0)
            running_hausdorff += hausdorff * img.size(0)
            n += img.size(0)
            iter_count += 1

            # 训练日志，含时间戳
            if iter_count % log_iter_interval == 0 or iter_count == len(loaders['train']):
                msg = (f"{timestamp_str()} Epoch {epoch}, Iter {iter_count}: "
                       f"Loss={loss.item():.4f}, Dice={dice:.4f}, IoU={iou:.4f}, Hausdorff={hausdorff:.4f}")
                print(msg)
                log_fp.write(msg + "\n")
                log_fp.flush()

        train_loss = running_loss / n
        train_dice = running_dice / n
        train_iou = running_iou / n
        train_hausdorff = running_hausdorff / n

        # 验证
        model_net.eval()
        val_loss = val_dice = val_iou = val_hausdorff = vn = 0
        with torch.no_grad():
            for batch in loaders['val']:
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                ssm = batch['ssm'].to(device)
                out = model_net(img, ssm)
                loss = criterion(out, lbl)
                val_loss += loss.item() * img.size(0)
                pred = torch.argmax(out, dim=1)
                dice = np.mean(dice_coef(pred, lbl))
                iou = iou_coef((pred > 0), (lbl > 0))
                hausdorff_metric.reset()
                hausdorff_metric(pred, lbl)
                hausdorff = hausdorff_metric.aggregate().item()
                val_dice += dice * img.size(0)
                val_iou += iou * img.size(0)
                val_hausdorff += hausdorff * img.size(0)
                vn += img.size(0)
        val_loss /= vn
        val_dice /= vn
        val_iou /= vn
        val_hausdorff /= vn

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['train_hausdorff'].append(train_hausdorff)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_hausdorff'].append(val_hausdorff)

        writer.add_scalar(f"{structure}/train/loss", train_loss, epoch)
        writer.add_scalar(f"{structure}/train/dice", train_dice, epoch)
        writer.add_scalar(f"{structure}/train/iou", train_iou, epoch)
        writer.add_scalar(f"{structure}/train/hausdorff", train_hausdorff, epoch)
        writer.add_scalar(f"{structure}/val/loss", val_loss, epoch)
        writer.add_scalar(f"{structure}/val/dice", val_dice, epoch)
        writer.add_scalar(f"{structure}/val/iou", val_iou, epoch)
        writer.add_scalar(f"{structure}/val/hausdorff", val_hausdorff, epoch)

        # epoch结尾输出
        msg = (f"{timestamp_str()} [{structure}] Epoch {epoch:3d}: "
               f"train_loss={train_loss:.4f}, dice={train_dice:.4f}, iou={train_iou:.4f}, hausdorff={train_hausdorff:.4f} | "
               f"val_loss={val_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}, hausdorff={val_hausdorff:.4f}")
        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

        # 手动保存checkpoint
        if (epoch + 1) % save_every_epoch == 0:
            torch.save(model_net.state_dict(), os.path.join(exp_dir, f"checkpoint_epoch{epoch+1}.pth"))
            print(f"{timestamp_str()} [INFO] Saved checkpoint at epoch {epoch+1}")

        # 手动检查/更新best model
        if (epoch + 1) % update_best_every_epoch == 0 and val_dice > best_dice:
            best_dice = val_dice
            best_w = copy.deepcopy(model_net.state_dict())
            torch.save(best_w, os.path.join(exp_dir, "checkpoint_best.pth"))
            print(f"{timestamp_str()} [INFO] Updated best model at epoch {epoch+1}")


    # ======= 测试阶段 =======
    model_net.load_state_dict(best_w)
    model_net.eval()
    test_loss = test_dice = test_iou = test_hausdorff = tn = 0
    with torch.no_grad():
        for batch in loaders['test']:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            ssm = batch['ssm'].to(device)
            out = model_net(img, ssm)
            loss = criterion(out, lbl)
            test_loss += loss.item() * img.size(0)
            pred = torch.argmax(out, dim=1)
            dice = np.mean(dice_coef(pred, lbl))
            iou = iou_coef((pred > 0), (lbl > 0))
            hausdorff_metric.reset()
            hausdorff_metric(pred, lbl)
            hausdorff = hausdorff_metric.aggregate().item()
            test_dice += dice * img.size(0)
            test_iou  += iou * img.size(0)
            test_hausdorff += hausdorff * img.size(0)
            tn += img.size(0)
    test_loss /= tn
    test_dice /= tn
    test_iou  /= tn
    test_hausdorff /= tn

    summary = f"""
    -------------------- Experiment Summary [{structure}] --------------------
    Total epochs           : {num_epochs}
    Batch size             : {batch_size}
    Iterations/epoch       : {len(loaders['train'])}
    Train images           : {len(train_set)}
    Latent dim             : {train_set.latent_df.shape[1]}
    Best Val Dice (max)    : {max(history['val_dice']):.4f}
    Final Test Dice        : {test_dice:.4f}
    Final Test IoU         : {test_iou:.4f}
    Final Test Hausdorff   : {test_hausdorff:.4f}
    Test Loss              : {test_loss:.4f}
    ----------------------------------------------------------
    """
    print(summary)
    log_fp.write(summary)
    log_fp.flush()
    writer.close()
    log_fp.close()
    sys.stdout = sys.__stdout__
    print(f"[INFO] {structure} Training/validation logs, weights and log files saved to: {exp_dir}")
