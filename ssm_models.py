import torch
import torch.nn as nn

# =========================
#      Basic UNet Blocks
# =========================

class ConvBlock3D(nn.Module):
    """
    A standard 3D UNet encoding/decoding block: 
    Two Conv3d + BatchNorm3d + ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock3D(nn.Module):
    """
    3D UNet upsampling block: 
    ConvTranspose3d for upsampling, skip-connection, and ConvBlock3D.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# =========================
#   SSM Fusion Mechanisms
# =========================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for Conditional Modulation
    ----------------------------------------------------------------
    Abstract:
      - Modulates the input feature map by predicting a per-channel affine transformation 
        (gamma * x + beta), where both gamma and beta are learned from the SSM latent vector.
      - Can be inserted at any feature map layer (e.g., encoder, bottleneck).
    Implementation:
      - Input: feature [B, C, D, H, W] and SSM latent [B, ssm_dim]
      - FC(SSM latent) -> 2*C scalars -> split to gamma, beta
      - Applies InstanceNorm3d (or other norm) to the feature map
      - Output: Modulated feature [B, C, D, H, W]
    Typical use:
      - At encoder-1 (early fusion, low-level features)
      - At bottleneck (late fusion, global context)
    """
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features)
        self.fc = nn.Linear(cond_dim, num_features * 2)
    def forward(self, x, cond):
        x = self.norm(x)
        gamma, beta = self.fc(cond).chunk(2, dim=1)
        gamma = gamma.reshape(-1, x.size(1), 1, 1, 1)
        beta = beta.reshape(-1, x.size(1), 1, 1, 1)
        return gamma * x + beta

class ConditionalSEBlock(nn.Module):
    """
    Conditional Squeeze-and-Excitation (SE) Attention
    -------------------------------------------------
    Abstract:
      - Predicts channel-wise scaling factors (attention) from the SSM latent.
      - Rescales feature map at a given layer (usually encoder-1).
      - Realizes conditional recalibration: SSM latent controls which channels to enhance/suppress.
    Implementation:
      - Input: feature [B, C, D, H, W] and SSM latent [B, ssm_dim]
      - FC(SSM latent) -> reduction -> relu -> expand -> sigmoid -> [B, C]
      - Expand to [B, C, 1, 1, 1] and multiply with input feature
    Typical use:
      - Applied at encoder-1 for global shape conditioning.
    """
    def __init__(self, channels, cond_dim, reduction=16):
        super().__init__()
        self.condition_fc = nn.Sequential(
            nn.Linear(cond_dim, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x, ssm):
        attention = self.condition_fc(ssm).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
        return x * attention

class AuxiliaryBranch(nn.Module):
    """
    Auxiliary MLP Fusion Branch at Bottleneck
    -----------------------------------------
    Abstract:
      - Transforms the SSM latent vector into a 3D feature map via an MLP, 
        then concatenates this auxiliary feature with the main bottleneck feature map,
        and fuses them using a 1x1 Conv3d.
      - Allows the network to inject prior/global shape information at the highest abstraction level.
    Implementation:
      - Input: bottleneck feature [B, C, D, H, W], SSM latent [B, ssm_dim]
      - MLP(SSM latent) -> [B, C]
      - Expand and broadcast to [B, C, D, H, W]
      - Concat along channel with bottleneck feature -> [B, 2C, D, H, W]
      - 1x1 Conv3d -> [B, C, D, H, W]
    Typical use:
      - Applied only at bottleneck, can be combined with other fusion modules.
    """
    def __init__(self, ssm_dim, out_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ssm_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )
        self.conv = nn.Conv3d(out_channels * 2, out_channels, 1)

    def forward(self, feat, ssm):
        # SSM feature: [B, out_channels, 1, 1, 1] broadcast to feat shape
        a = self.fc(ssm)[:, :, None, None, None].expand_as(feat)
        return self.conv(torch.cat([feat, a], dim=1))

# =========================
#      Main Model
# =========================

class UNet3DWithSSM(nn.Module):
    """
    UNet3DWithSSM: 3D UNet with Configurable SSM (Statistical Shape Model) Fusion Strategies
    ---------------------------------------------------------------------------------------
    Abstract:
      - Standard 3D UNet backbone with optional shape prior fusion at multiple levels.
      - Four main fusion strategies are supported via boolean flags:
        1. baseline:   No SSM fusion. Pure UNet (for ablation).
        2. attn:       SSM latent fused via Conditional SE (attention) at encoder-1.
        3. aux:        SSM latent fused via auxiliary MLP branch at bottleneck.
        4. film:       SSM latent fused via FiLM modulation at encoder-1 and bottleneck.
      - These can be combined as needed for research or ablation studies.
    Implementation:
      - All SSM fusion modules are controlled by `use_film`, `use_aux`, and `use_attention`.
      - SSM latent input: shape [B, ssm_dim] (dataloader自动对齐)
      - Forward pass applies the selected SSM fusion module(s) at the appropriate layers.
    """
    def __init__(self, in_ch, base_ch, ssm_dim, num_classes=1,
                 use_film=False, use_aux=False, use_attention=False):
        super().__init__()
        self.use_film = use_film
        self.use_aux = use_aux
        self.use_attention = use_attention

        # ---------- Encoder ----------
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base_ch * 4, base_ch * 8)

        # ---------- Decoder ----------
        self.up3 = UpBlock3D(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock3D(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock3D(base_ch * 2, base_ch)
        self.final = nn.Conv3d(base_ch, num_classes, 1)

        # ---------- Conditional SSM Fusion Modules ----------
        # Each fusion module can be independently enabled for experiments
        if self.use_film:
            # FiLM modules for encoder-1 and bottleneck
            self.film1 = FiLM(base_ch, ssm_dim)
            self.film3 = FiLM(base_ch * 8, ssm_dim)
        if self.use_attention:
            # Conditional SE attention for encoder-1
            self.att1 = ConditionalSEBlock(base_ch, ssm_dim)
        if self.use_aux:
            # Auxiliary branch at bottleneck
            self.aux = AuxiliaryBranch(ssm_dim, base_ch * 8)

    def forward(self, x, ssm):
        """
        Forward Pass
        x   : [B, 1, D, H, W]   -- input volume
        ssm : [B, ssm_dim]      -- SSM latent vector
        """
        # ---- Encoder 1 ----
        x1 = self.enc1(x)              # [B, base_ch, D, H, W]
        # Optionally fuse SSM at encoder-1 (low-level features)
        if self.use_film:
            x1 = self.film1(x1, ssm)
        if self.use_attention:
            x1 = self.att1(x1, ssm)

        # ---- Encoder 2, 3 ----
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # ---- Bottleneck ----
        b = self.bottleneck(self.pool(x3))
        # Optionally fuse SSM at bottleneck (high-level features)
        if self.use_aux:
            b = self.aux(b, ssm)
        if self.use_film:
            b = self.film3(b, ssm)

        # ---- Decoder ----
        u3 = self.up3(b, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)

        out = self.final(u1)  # [B, num_classes, D, H, W]
        # For binary segmentation: usually use sigmoid outside
        # For multi-class segmentation: use softmax outside
        return out
