# DeepSSM-Fusion-Model
This repository implements a 3D UNet-based segmentation framework enhanced with dual statistical shape priors, derived from the DeepSSM family of models.
Specifically, the method fuses epicardial and endocardial latent shape embeddings—extracted by TL-DeepSSM (image encoder) and Base-DeepSSM (label encoder)—into the UNet via FiLM, attention, and auxiliary branch mechanisms.

The project supports:

End-to-end training of UNet3D with dual latent shape priors

Configurable fusion mechanisms (FiLM, attention, auxiliary, concat, full)

Automatic dataset splitting + latent extraction

Flexible training/testing pipelines

Slice visualization + NIfTI saving

Per-case evaluation metrics (Dice / Loss)

This is a complete, reproducible pipeline for 3D cardiac segmentation with shape constraints.

🚀 1. Project Overview

Traditional UNet segmentation lacks global anatomical constraints, often causing:
holes / missing myocardium
irregular geometry
inconsistent structure across slices
To address this, we incorporate global cardiac priors from DeepSSM:
Structure	Encoder	Type	Used for
Epicardium	TL-DeepSSM	Image encoder	Bottleneck modulation
Endocardium	Base-DeepSSM	Label encoder	Encoder modulation

The two latent codes (ssm_epi, ssm_endo) are fused into the UNet in multiple ways.

2. Fusion UNet Architecture

The core model is implemented in train_ssm_fusion.py .

Supported fusion strategies
Mode	Encoder Fusion	Bottleneck Fusion
baseline	×	×
attn	Conditional SE attention (endo)	×
aux	×	Auxiliary branch (epi)
film	FiLM (endo, epi)	FiLM (epi)
concat	Concatenate latent vectors	×
full	FiLM + Attention + Auxiliary	✔✔✔
Model diagram
Input → Encoder1 —(FiLM/Attn with ssm_endo)—→ Encoder2 → Encoder3
                ↓
         Bottleneck —(Aux/FiLM with ssm_epi)—→ Decoder → Output Mask

UNet backbone uses:
3D convolution blocks
Skip connections
Final sigmoid output for binary myocardium segmentation

3. Repository Structure
.
├── train_ssm_fusion.py               # Training pipeline with fusion options
├── test_ssm_fusion.py                # Evaluation + visualization
├── dataloader.py                     # 3D MRI + ssm_epi/ssm_endo loader
├── MiteaSegmentationFusionModel.py   # (Older version)
├── MiteaSegmentationFusionModelEpi.py# (TL-DeepSSM fusion model)
├── seg_training.py / seg_testing.py  # UNet baseline training/testing
├── tools/
│   └── ShapeWorks latent extraction utilities
└── data/
    ├── MITEA/
    │   ├── images/
    │   ├── labels/
    │   ├── split_train.txt
    │   ├── split_val.txt
    │   ├── split_test.txt
    │   └── latent_ssm.csv (auto-generated)


Latent extraction script: train_ssm_fusion.py and ssm_extraction.py (part of your files).
Dataset split & latent generator: extract_latent_ssm.py .

4. Data Preparation

4.1 Original Dataset Structure
MITEA/
├── images/*.nii.gz
└── labels/*.nii.gz

4.2 Automatic dataset splitting

Run:
python extract_latent_ssm.py
This will:
✔ Create train/ val/ test/ folders
✔ Generate split_*.txt lists
✔ Load TL-DeepSSM & Base-DeepSSM encoders
✔ Extract per-image latent codes
✔ Save latent_ssm.csv
Example output:
latent_ssm.csv: N × (latent_epi + latent_endo)

5. Dual Shape Prior Extraction

Latent extraction code from:
Epicardial (TL-DeepSSM image encoder) → epi_latent
Endocardial (Base-DeepSSM label encoder) → endo_latent
Implementation reference from:
MiteaSegmentationFusionModelEpi.py
extract_latent_ssm.py
These encoders are frozen during segmentation model training.

6. Training UNet3D With Shape Fusion

Command
python train_ssm_fusion.py
You can edit these lines to enable specific fusion structures:
selected_names = {"baseline", "attn", "aux", "film"}   # choose modes to train
Training loop logs:
train loss / dice
val loss / dice
best checkpoint every 5 epochs
full checkpoint every 10 epochs
All logs saved to:

experiments/<mode_timestamp>/log/

7. Testing & Metrics

Run:
python test_ssm_fusion.py \
    --mode film \
    --checkpoint path/to/checkpoint.pth \
    --data_root data/ED \
    --save_dir predictions/film_test
This script performs:
✔ Case-wise Dice & Loss
✔ Save metrics.csv
✔ Compute mean ± std
✔ Save NIfTI predictions
✔ Generate slice visualization
Example metrics:
Dice: 0.8893 ± 0.0341
Loss: 0.1214 ± 0.0065
Implementation: test_ssm_fusion.py .

8. Visualization Output

Each test case generates:
Axial / Coronal / Sagittal comparison
Contours: GT (red) vs Pred (blue)
Raw prediction .nii.gz
Example:
predictions/film_test/
├── 0_MITEA_001.png
├── 0_MITEA_001_pred.nii.gz
└── metrics.csv

9. Baseline UNet (No Shape Prior)

Included for comparison:
seg_training.py
seg_testing.py
Supports Dice, Hausdorff distance, Mean IoU.

10. Key Features Summary

Feature	Supported
3D UNet segmentation	✔
Dual shape priors (epi + endo)	✔
TL-DeepSSM image encoder	✔
Base-DeepSSM label encoder	✔
FiLM fusion	✔
Conditional attention	✔
Auxiliary latent branch	✔
Multi-mode training	✔
Automatic dataset splitting	✔
Latent extraction via ShapeWorks	✔
NIfTI prediction saving	✔
Visualization & contouring	✔
Per-case metrics CSV	✔

11. Dependencies

Python 3.9+
PyTorch
PyTorch Lightning
MONAI
ShapeWorks 6.6.1
nibabel, scipy, pandas, matplotlib


12. Acknowledgements

This project integrates:
ShapeWorks DeepSSM pipelines
MONAI 3D segmentation framework
TUM CAMP research environment