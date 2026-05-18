"""
=============================================================================
  YOLOv26n — Architectural Component Ablation Study
=============================================================================
  Systematically removes / replaces architectural modules to measure their
  individual contribution to detection performance.

  Variants:
    A1 — Full Model        : Attention-C2f + PAN-FPN + Decoupled Head
    A2 — Without Attention : Standard C2f  + PAN-FPN + Decoupled Head
    A3 — Without PAN       : Standard C2f  + FPN only + Decoupled Head
    A4 — Without FPN       : Standard C2f  + PAN only + Decoupled Head
    A5 — Coupled Head      : Attention-C2f + PAN-FPN  + Coupled Head
    A6 — Without Att + PAN : Standard C2f  + FPN only + Decoupled Head

  What each component does:
    Attention-C2f : C2f bottleneck augmented with CBAM channel+spatial attention.
                    Helps the model focus on discriminative spatial regions.
    PAN-FPN       : Path Aggregation Network + Feature Pyramid Network.
                    PAN adds bottom-up path to FPN for better small-obj recall.
    Decoupled Head: Separate branches for classification and box regression.
                    Reduces task conflict vs a single shared coupled head.

  Requirements:
    pip install ultralytics torch torchvision numpy pandas matplotlib
=============================================================================
"""

import csv
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("arch-ablation")


# =============================================================================
#  SECTION 1 — Configuration
# =============================================================================

@dataclass
class ArchConfig:
    """Edit paths before running."""

    # ── Model ─────────────────────────────────────────────────────────────────
    model_weights   : str   = r"E:\YOLOV8\best26n.pt"
    data_yaml       : str   = r"C:\Users\Asus\YOLO26\ISL YOLO\data.yaml"
    project_dir     : str   = "runs/arch_ablation"
    results_csv     : str   = "arch_ablation_results.csv"

    # ── Training ──────────────────────────────────────────────────────────────
    epochs          : int   = 100
    imgsz           : int   = 640
    batch           : int   = 32
    device          : str   = "0"          # RTX 3060
    workers         : int   = 0            # Windows safe
    seed            : int   = 42
    patience        : int   = 30

    # ── Loss weights ──────────────────────────────────────────────────────────
    box_gain        : float = 7.5
    cls_gain        : float = 0.5
    dfl_gain        : float = 1.5

    # ── Architecture ──────────────────────────────────────────────────────────
    num_classes     : int   = 25           # ISL dataset
    width_multiple  : float = 0.25         # YOLOv26n width multiplier

    # ── Attention (CBAM) ──────────────────────────────────────────────────────
    attn_reduction  : int   = 16           # channel reduction ratio in SE
    attn_kernel     : int   = 7            # spatial attention kernel size

    # ── Head ──────────────────────────────────────────────────────────────────
    reg_max         : int   = 16           # DFL bins


cfg = ArchConfig()


# =============================================================================
#  SECTION 2 — Building Blocks
# =============================================================================

def autopad(k, p=None, d=1):
    """Auto-compute padding to maintain spatial size."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Standard conv → BN → SiLU. No groups parameter (use nn.Conv2d directly)."""
    def __init__(self, c_in, c_out, k=1, s=1, p=None, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, k, s, autopad(k, p, d), dilation=d, bias=False
        )
        self.bn  = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard YOLOv26n bottleneck with optional shortcut.
    Groups (g) passed directly to nn.Conv2d — not through Conv wrapper.
    Fixed: Conv wrapper does not accept 'g' keyword argument.
    """
    def __init__(self, c_in, c_out, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = Conv(c_in, c_hidden, 3, 1)
        # Pass groups directly to nn.Conv2d — Conv wrapper has no groups param
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_hidden, c_out, 3, 1, 1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
        self.add = shortcut and (c_in == c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ─────────────────────────────────────────────────────────────────────────────
#  2a. Channel & Spatial Attention (CBAM)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-Excitation channel attention.
    Globally pools spatial dims → learns channel-wise recalibration weights.
    Suppresses uninformative feature channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """
    Spatial attention — highlights which locations to focus on
    by pooling across channels then convolving.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg    = x.mean(dim=1, keepdim=True)
        mx, _  = x.max(dim=1, keepdim=True)
        scale  = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention then spatial attention sequentially.
    Used inside AttentionC2f to enrich feature representation.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────────────────────────────────────────
#  2b. C2f variants — Standard and Attention-augmented
# ─────────────────────────────────────────────────────────────────────────────

class C2f(nn.Module):
    """
    Standard C2f module (Cross-Stage Partial with 2 convolutions).
    Splits channels, processes through n bottlenecks, concatenates.
    Fix: g is not passed to Conv — Bottleneck handles groups internally.
    """
    def __init__(self, c_in, c_out, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c   = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c_out, 1)
        # Fixed: pass g=1 (standard convolutions) — no keyword to Conv wrapper
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=1, e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class AttentionC2f(nn.Module):
    """
    Attention-augmented C2f (used in Full Model A1 and Coupled Head A5).
    Identical to C2f but with CBAM applied after the final concat.

    Why it helps: CBAM recalibrates feature maps so the network attends to
    object-relevant channels and spatial regions, suppressing background.
    Particularly useful for small ISL hand gestures.
    Fix: g is not passed to Conv — Bottleneck handles groups internally.
    """
    def __init__(self, c_in, c_out, n=1, shortcut=False, g=1, e=0.5,
                 reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.c   = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c_out, 1)
        # Fixed: pass g=1 (standard convolutions)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=1, e=1.0)
            for _ in range(n)
        )
        self.attn = CBAM(c_out, reduction, kernel)   # CBAM after merge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, dim=1))
        return self.attn(out)                         # attention on output


# ─────────────────────────────────────────────────────────────────────────────
#  2c. Neck — PAN-FPN, FPN-only, PAN-only
# ─────────────────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) — top-down only.
    Fuses high-level semantics downward to lower-resolution feature maps.
    Good for large objects; less effective on small objects.
    Used in: A3 (Without PAN), A6 (Without Att + PAN)
    """
    def __init__(self, channels: List[int], use_attention: bool = False,
                 cfg: ArchConfig = None):
        super().__init__()
        cfg      = cfg or ArchConfig()
        C2fCls   = AttentionC2f if use_attention else C2f
        reduction = cfg.attn_reduction
        kernel    = cfg.attn_kernel

        self.lat_p4   = Conv(channels[2], channels[1], 1)
        self.lat_p3   = Conv(channels[1], channels[0], 1)

        if use_attention:
            self.c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1,
                                  reduction=reduction, kernel=kernel)
            self.c2f_p3 = C2fCls(channels[0]*2, channels[0], n=1,
                                  reduction=reduction, kernel=kernel)
        else:
            self.c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1)
            self.c2f_p3 = C2fCls(channels[0]*2, channels[0], n=1)

    def forward(self, features: Tuple[torch.Tensor, ...]):
        c3, c4, c5 = features
        p4 = self.c2f_p4(torch.cat([
            F.interpolate(self.lat_p4(c5), size=c4.shape[2:], mode="nearest"),
            c4
        ], dim=1))
        p3 = self.c2f_p3(torch.cat([
            F.interpolate(self.lat_p3(p4), size=c3.shape[2:], mode="nearest"),
            c3
        ], dim=1))
        return p3, p4, c5


class PANNeck(nn.Module):
    """
    Path Aggregation Network (PAN) — bottom-up only.
    Propagates fine-grained low-level features upward.
    Better at small objects than FPN alone.
    Used in: A4 (Without FPN)
    """
    def __init__(self, channels: List[int], use_attention: bool = False,
                 cfg: ArchConfig = None):
        super().__init__()
        cfg      = cfg or ArchConfig()
        C2fCls   = AttentionC2f if use_attention else C2f
        reduction = cfg.attn_reduction
        kernel    = cfg.attn_kernel

        self.down_p3 = Conv(channels[0], channels[1], 3, 2)
        self.down_p4 = Conv(channels[1], channels[2], 3, 2)

        if use_attention:
            self.c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1,
                                  reduction=reduction, kernel=kernel)
            self.c2f_p5 = C2fCls(channels[2]*2, channels[2], n=1,
                                  reduction=reduction, kernel=kernel)
        else:
            self.c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1)
            self.c2f_p5 = C2fCls(channels[2]*2, channels[2], n=1)

    def forward(self, features: Tuple[torch.Tensor, ...]):
        c3, c4, c5 = features
        p4 = self.c2f_p4(torch.cat([self.down_p3(c3), c4], dim=1))
        p5 = self.c2f_p5(torch.cat([self.down_p4(p4), c5], dim=1))
        return c3, p4, p5


class PANFPNNeck(nn.Module):
    """
    Full PAN-FPN neck (used in A1 Full Model, A2 No-Attention, A5 Coupled).

    Two-stage bidirectional fusion:
      Stage 1 (FPN): top-down  P5 → P4 → P3  (semantic enrichment)
      Stage 2 (PAN): bottom-up P3 → P4 → P5  (detail propagation)

    Bidirectional design outperforms FPN-only or PAN-only on multi-scale objects.
    """
    def __init__(self, channels: List[int], use_attention: bool = False,
                 cfg: ArchConfig = None):
        super().__init__()
        cfg      = cfg or ArchConfig()
        C2fCls   = AttentionC2f if use_attention else C2f
        reduction = cfg.attn_reduction
        kernel    = cfg.attn_kernel

        # ── FPN top-down ──────────────────────────────────────────────────────
        self.lat_p4     = Conv(channels[2], channels[1], 1)
        self.lat_p3     = Conv(channels[1], channels[0], 1)

        if use_attention:
            self.fpn_c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1,
                                      reduction=reduction, kernel=kernel)
            self.fpn_c2f_p3 = C2fCls(channels[0]*2, channels[0], n=1,
                                      reduction=reduction, kernel=kernel)
            self.pan_c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1,
                                      reduction=reduction, kernel=kernel)
            self.pan_c2f_p5 = C2fCls(channels[2]*2, channels[2], n=1,
                                      reduction=reduction, kernel=kernel)
        else:
            self.fpn_c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1)
            self.fpn_c2f_p3 = C2fCls(channels[0]*2, channels[0], n=1)
            self.pan_c2f_p4 = C2fCls(channels[1]*2, channels[1], n=1)
            self.pan_c2f_p5 = C2fCls(channels[2]*2, channels[2], n=1)

        # ── PAN bottom-up ─────────────────────────────────────────────────────
        self.down_p3 = Conv(channels[0], channels[1], 3, 2)
        self.down_p4 = Conv(channels[1], channels[2], 3, 2)

    def forward(self, features: Tuple[torch.Tensor, ...]):
        c3, c4, c5 = features

        # Stage 1: FPN top-down
        p4_td = self.fpn_c2f_p4(torch.cat([
            F.interpolate(self.lat_p4(c5), size=c4.shape[2:], mode="nearest"),
            c4
        ], dim=1))
        p3_td = self.fpn_c2f_p3(torch.cat([
            F.interpolate(self.lat_p3(p4_td), size=c3.shape[2:], mode="nearest"),
            c3
        ], dim=1))

        # Stage 2: PAN bottom-up
        p4_bu = self.pan_c2f_p4(torch.cat([self.down_p3(p3_td), p4_td], dim=1))
        p5_bu = self.pan_c2f_p5(torch.cat([self.down_p4(p4_bu), c5],    dim=1))

        return p3_td, p4_bu, p5_bu


# ─────────────────────────────────────────────────────────────────────────────
#  2d. Detection Head — Decoupled vs Coupled
# ─────────────────────────────────────────────────────────────────────────────

class DecoupledHead(nn.Module):
    """
    Decoupled detection head.
    Separate conv branches for classification and regression.
    Avoids gradient conflicts → faster convergence, better accuracy.
    Used in: A1, A2, A3, A4, A6
    """
    def __init__(self, num_classes: int, in_channels: int, reg_max: int = 16):
        super().__init__()
        self.nc      = num_classes
        self.reg_max = reg_max
        mid          = max(in_channels, num_classes, 16)

        self.cls_conv = nn.Sequential(Conv(in_channels, mid, 3),
                                      Conv(mid, mid, 3))
        self.cls_pred = nn.Conv2d(mid, num_classes, 1)

        self.reg_conv = nn.Sequential(Conv(in_channels, mid, 3),
                                      Conv(mid, mid, 3))
        self.reg_pred = nn.Conv2d(mid, 4 * reg_max, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls_pred(self.cls_conv(x))
        reg = self.reg_pred(self.reg_conv(x))
        return torch.cat([reg, cls], dim=1)


class CoupledHead(nn.Module):
    """
    Coupled detection head.
    Shared conv layers for both cls and reg — simpler but task interference
    can degrade accuracy especially for similar classes.
    Used in: A5
    """
    def __init__(self, num_classes: int, in_channels: int, reg_max: int = 16):
        super().__init__()
        self.nc      = num_classes
        self.reg_max = reg_max
        mid          = max(in_channels, num_classes, 16)

        self.shared = nn.Sequential(Conv(in_channels, mid, 3),
                                    Conv(mid, mid, 3))
        self.pred   = nn.Conv2d(mid, num_classes + 4 * reg_max, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.shared(x))


# ─────────────────────────────────────────────────────────────────────────────
#  2e. Backbone (shared across all variants)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv26nBackbone(nn.Module):
    """
    Lightweight YOLOv26n backbone.
    Produces P3, P4, P5 feature maps at strides 8, 16, 32.
    Shared across all 6 ablation variants — only neck and head differ.

    Channel widths with width_multiple=0.25:
      P3: 32ch  P4: 64ch  P5: 64ch
    """
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        w  = cfg.width_multiple
        ch = lambda x: max(1, int(x * w))

        # Stem
        self.stem = nn.Sequential(
            Conv(3,      ch(32),  3, 2),
            Conv(ch(32), ch(64),  3, 2),
            C2f(ch(64),  ch(64),  n=1, shortcut=True),
        )
        # P3 (stride 8)
        self.stage1 = nn.Sequential(
            Conv(ch(64),  ch(128), 3, 2),
            C2f(ch(128),  ch(128), n=2, shortcut=True),
        )
        # P4 (stride 16)
        self.stage2 = nn.Sequential(
            Conv(ch(128), ch(256), 3, 2),
            C2f(ch(256),  ch(256), n=2, shortcut=True),
        )
        # P5 (stride 32) — compressed to ch(256)
        self.stage3 = nn.Sequential(
            Conv(ch(256), ch(512), 3, 2),
            C2f(ch(512),  ch(512), n=1, shortcut=True),
            Conv(ch(512), ch(256), 1),
        )

        self.out_channels = [ch(128), ch(256), ch(256)]

    def forward(self, x: torch.Tensor):
        x  = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return p3, p4, p5


# =============================================================================
#  SECTION 3 — Full Detector Assembly
# =============================================================================

class YOLOv26nDetector(nn.Module):
    """Assembles backbone + neck + head. Neck and head are swappable."""
    def __init__(self, backbone, neck, heads, strides=None):
        super().__init__()
        self.backbone = backbone
        self.neck     = neck
        self.heads    = heads
        self.strides  = strides or [8, 16, 32]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features    = self.backbone(x)
        p3, p4, p5  = self.neck(features)
        return [h(f) for h, f in zip(self.heads, [p3, p4, p5])]


def build_detector(
    neck_type    : str       = "panfpn",
    use_attention: bool      = True,
    head_type    : str       = "decoupled",
    cfg          : ArchConfig = None,
) -> YOLOv26nDetector:
    """
    Factory — builds one YOLOv26nDetector for any ablation variant.
    neck_type    : 'panfpn' | 'fpn' | 'pan'
    use_attention: True → AttentionC2f, False → standard C2f
    head_type    : 'decoupled' | 'coupled'
    """
    cfg      = cfg or ArchConfig()
    backbone = YOLOv26nBackbone(cfg)
    ch       = backbone.out_channels

    neck_map = {
        "panfpn": PANFPNNeck,
        "fpn"   : FPNNeck,
        "pan"   : PANNeck,
    }
    if neck_type not in neck_map:
        raise ValueError(f"neck_type must be one of {list(neck_map.keys())}")

    neck  = neck_map[neck_type](ch, use_attention=use_attention, cfg=cfg)
    HCls  = DecoupledHead if head_type == "decoupled" else CoupledHead
    heads = nn.ModuleList([HCls(cfg.num_classes, c, cfg.reg_max) for c in ch])

    return YOLOv26nDetector(backbone, neck, heads)


# =============================================================================
#  SECTION 4 — Six Ablation Variant Definitions
# =============================================================================

@dataclass
class VariantSpec:
    name          : str
    label         : str
    description   : str
    neck_type     : str
    use_attention : bool
    head_type     : str


ABLATION_VARIANTS: List[VariantSpec] = [
    VariantSpec(
        name="A1_Full",          label="Full Model",
        description="Attention-C2f + PAN-FPN + Decoupled Head",
        neck_type="panfpn",      use_attention=True,  head_type="decoupled",
    ),
    VariantSpec(
        name="A2_NoAttention",   label="Without Attention",
        description="Standard C2f + PAN-FPN + Decoupled Head",
        neck_type="panfpn",      use_attention=False, head_type="decoupled",
    ),
    VariantSpec(
        name="A3_NoPAN",         label="Without PAN",
        description="Standard C2f + FPN only + Decoupled Head",
        neck_type="fpn",         use_attention=False, head_type="decoupled",
    ),
    VariantSpec(
        name="A4_NoFPN",         label="Without FPN",
        description="Standard C2f + PAN only + Decoupled Head",
        neck_type="pan",         use_attention=False, head_type="decoupled",
    ),
    VariantSpec(
        name="A5_CoupledHead",   label="Coupled Head",
        description="Attention-C2f + PAN-FPN + Coupled Head",
        neck_type="panfpn",      use_attention=True,  head_type="coupled",
    ),
    VariantSpec(
        name="A6_NoAttnNoPAN",   label="Without Attention + PAN",
        description="Standard C2f + FPN only + Decoupled Head",
        neck_type="fpn",         use_attention=False, head_type="decoupled",
    ),
]


# =============================================================================
#  SECTION 5 — Parameter & FLOP Counter
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, imgsz: int = 640) -> float:
    try:
        from thop import profile  # type: ignore
        dummy        = torch.zeros(1, 3, imgsz, imgsz)
        flops, _     = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return count_parameters(model) * 2 * (imgsz ** 2) / 1e9


def model_size_mb(model: nn.Module) -> float:
    return count_parameters(model) * 4 / 1e6


# =============================================================================
#  SECTION 6 — Ablation Trainer
# =============================================================================

class ArchVariantTrainer:
    """Trains one architectural variant via Ultralytics YOLO."""

    def __init__(self, spec: VariantSpec, cfg: ArchConfig):
        self.spec   = spec
        self.cfg    = cfg
        # Build custom model for param/FLOP counting only
        self._model = build_detector(
            neck_type     = spec.neck_type,
            use_attention = spec.use_attention,
            head_type     = spec.head_type,
            cfg           = cfg,
        )

    @property
    def params(self) -> int:
        return count_parameters(self._model)

    @property
    def gflops(self) -> float:
        return count_flops(self._model, self.cfg.imgsz)

    @property
    def size_mb(self) -> float:
        return model_size_mb(self._model)

    def train(self) -> dict:
        """Full training run. Returns metrics dict."""
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not installed: pip install ultralytics")
            return {}

        log.info("═" * 65)
        log.info(f"  {self.spec.name}: {self.spec.description}")
        log.info(f"  Params={self.params:,}  GFLOPs={self.gflops:.2f}  "
                 f"Size={self.size_mb:.1f}MB")
        log.info("═" * 65)

        model = YOLO(self.cfg.model_weights)
        spec  = self.spec

        def on_train_start(trainer):
            log.info(f"Training variant: {spec.name}")
        model.add_callback("on_train_start", on_train_start)

        t0 = time.perf_counter()
        results = model.train(
            data         = self.cfg.data_yaml,
            epochs       = self.cfg.epochs,
            imgsz        = self.cfg.imgsz,
            batch        = self.cfg.batch,
            device       = self.cfg.device,
            workers      = self.cfg.workers,
            seed         = self.cfg.seed,
            patience     = self.cfg.patience,
            project      = self.cfg.project_dir,
            name         = self.spec.name,
            exist_ok     = True,
            box          = self.cfg.box_gain,
            cls          = self.cfg.cls_gain,
            dfl          = self.cfg.dfl_gain,
            close_mosaic = 10,
            verbose      = False,
        )
        elapsed = time.perf_counter() - t0

        metrics = model.metrics if hasattr(model, "metrics") else {}
        box_obj = getattr(metrics, "box", None)
        map50   = float(getattr(box_obj, "map50", 0) or 0)
        map5095 = float(getattr(box_obj, "map",   0) or 0)
        prec    = float(getattr(box_obj, "mp",    0) or 0)
        recall  = float(getattr(box_obj, "mr",    0) or 0)
        box_loss = float(
            getattr(results, "results_dict", {}).get("val/box_loss", 0)
        )

        log.info(
            f"  {self.spec.name} → mAP50={map50:.4f}  "
            f"mAP50-95={map5095:.4f}  P={prec:.4f}  R={recall:.4f}  "
            f"time={elapsed/60:.1f}min"
        )

        return {
            "variant"        : self.spec.name,
            "label"          : self.spec.label,
            "description"    : self.spec.description,
            "map50"          : map50,
            "map5095"        : map5095,
            "precision"      : prec,
            "recall"         : recall,
            "box_loss"       : box_loss,
            "params"         : self.params,
            "gflops"         : round(self.gflops, 3),
            "size_mb"        : round(self.size_mb, 2),
            "train_time_min" : round(elapsed / 60, 2),
        }

    def dry_run(self) -> dict:
        """Reports params, GFLOPs, size — no training."""
        log.info(f"  {self.spec.name}: {self.spec.description}")
        log.info(f"    Params   : {self.params:,}")
        log.info(f"    GFLOPs   : {self.gflops:.3f}")
        log.info(f"    Size     : {self.size_mb:.2f} MB")
        log.info(f"    Neck     : {self.spec.neck_type.upper()}")
        log.info(f"    Attention: {self.spec.use_attention}")
        log.info(f"    Head     : {self.spec.head_type}")
        return {
            "variant"    : self.spec.name,
            "label"      : self.spec.label,
            "description": self.spec.description,
            "params"     : self.params,
            "gflops"     : round(self.gflops, 3),
            "size_mb"    : round(self.size_mb, 2),
            "map50"      : 0.0,
            "map5095"    : 0.0,
            "precision"  : 0.0,
            "recall"     : 0.0,
            "box_loss"   : 0.0,
            "train_time_min": 0.0,
        }


# =============================================================================
#  SECTION 7 — Ablation Study Orchestrator
# =============================================================================

@dataclass
class ArchResult:
    variant         : str
    label           : str
    description     : str
    map50           : float = 0.0
    map5095         : float = 0.0
    precision       : float = 0.0
    recall          : float = 0.0
    box_loss        : float = 0.0
    params          : int   = 0
    gflops          : float = 0.0
    size_mb         : float = 0.0
    train_time_min  : float = 0.0
    map50_delta     : float = 0.0
    param_delta     : int   = 0


class ArchAblationStudy:
    """
    Runs all 6 architectural variants and produces:
      • Console summary table (accuracy + efficiency + contribution)
      • CSV with all metrics
      • 4-panel bar chart
      • Component contribution horizontal bar chart
    """

    def __init__(self, config: Optional[ArchConfig] = None):
        self.cfg     = config or ArchConfig()
        self.results : List[ArchResult] = []
        os.makedirs(self.cfg.project_dir, exist_ok=True)

    def run_all(self, dry_run: bool = False) -> List[ArchResult]:
        self.results = []
        for spec in ABLATION_VARIANTS:
            trainer = ArchVariantTrainer(spec, self.cfg)
            raw     = trainer.dry_run() if dry_run else trainer.train()
            self.results.append(ArchResult(**{
                k: raw.get(k, 0)
                for k in ArchResult.__dataclass_fields__
                if k not in ("map50_delta", "param_delta")
            }))
        self._compute_deltas()
        return self.results

    def run_variant(
        self, variant_name: str, dry_run: bool = False
    ) -> ArchResult:
        spec_map = {s.name: s for s in ABLATION_VARIANTS}
        if variant_name not in spec_map:
            raise ValueError(
                f"Unknown variant '{variant_name}'. "
                f"Choose from: {list(spec_map.keys())}"
            )
        trainer = ArchVariantTrainer(spec_map[variant_name], self.cfg)
        raw     = trainer.dry_run() if dry_run else trainer.train()
        result  = ArchResult(**{
            k: raw.get(k, 0)
            for k in ArchResult.__dataclass_fields__
            if k not in ("map50_delta", "param_delta")
        })
        self.results.append(result)
        self._compute_deltas()
        return result

    def _compute_deltas(self) -> None:
        """Compute mAP50 and param deltas vs A1 Full Model."""
        res_map     = {r.variant: r for r in self.results}
        ref         = res_map.get("A1_Full")
        ref_map50   = ref.map50  if ref else 0.0
        ref_params  = ref.params if ref else 0
        for r in self.results:
            r.map50_delta = r.map50  - ref_map50
            r.param_delta = r.params - ref_params

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if not self.results:
            log.warning("No results. Call run_all() first.")
            return

        sep = "─" * 118
        print(f"\n{'═'*118}")
        print(f"  YOLOv26n — Architectural Component Ablation Study")
        print(f"{'═'*118}")

        # Accuracy
        print(f"\n  ACCURACY")
        print(f"  {'Variant':<22} {'Description':<46} "
              f"{'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'ΔmAP50':>9}")
        print(f"  {sep}")
        for r in self.results:
            d = f"{r.map50_delta:+.4f}" if r.variant != "A1_Full" else "(ref)"
            print(
                f"  {r.variant:<22} {r.description:<46} "
                f"{r.map50:>7.4f} {r.map5095:>9.4f} "
                f"{r.precision:>7.4f} {r.recall:>7.4f} {d:>9}"
            )

        # Efficiency
        print(f"\n  EFFICIENCY")
        print(f"  {'Variant':<22} {'Params':>10} {'ΔParams':>10} "
              f"{'GFLOPs':>8} {'Size(MB)':>9} {'Time(min)':>10}")
        print(f"  {sep}")
        for r in self.results:
            dp = f"{r.param_delta:+,}" if r.variant != "A1_Full" else "(ref)"
            print(
                f"  {r.variant:<22} {r.params:>10,} {dp:>10} "
                f"{r.gflops:>8.3f} {r.size_mb:>9.2f} "
                f"{r.train_time_min:>10.1f}"
            )

        # Component contribution
        if any(r.map50 > 0 for r in self.results):
            print(f"\n  COMPONENT CONTRIBUTION (mAP50 drop when removed)")
            print(f"  {'Component':<25} {'Removed in':<24} {'mAP50 Drop':>12}")
            print(f"  {'─'*62}")
            res_map  = {r.variant: r for r in self.results}
            contrib  = {
                "Attention (CBAM)"  : ("A1_Full", "A2_NoAttention"),
                "PAN pathway"       : ("A1_Full", "A3_NoPAN"),
                "FPN pathway"       : ("A1_Full", "A4_NoFPN"),
                "Decoupled Head"    : ("A1_Full", "A5_CoupledHead"),
                "Attention + PAN"   : ("A1_Full", "A6_NoAttnNoPAN"),
            }
            for comp, (ref_k, abl_k) in contrib.items():
                if ref_k in res_map and abl_k in res_map:
                    drop = res_map[ref_k].map50 - res_map[abl_k].map50
                    print(f"  {comp:<25} {abl_k:<24} {drop:>+12.4f}")

        print(f"\n{'═'*118}\n")

    def save_csv(self, path: Optional[str] = None) -> str:
        path   = path or self.cfg.results_csv
        fields = [
            "variant","label","description",
            "map50","map5095","precision","recall","box_loss",
            "params","gflops","size_mb","train_time_min",
            "map50_delta","param_delta",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: getattr(r, k) for k in fields})
        log.info(f"Results saved → {path}")
        return path

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """4-panel bar chart: mAP50, mAP50-95, Params, GFLOPs."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed.")
            return
        if not self.results:
            return

        save_path = save_path or os.path.join(
            self.cfg.project_dir, "arch_ablation_plot.png"
        )
        labels = [r.variant.replace("_", "\n") for r in self.results]
        colors = ["#2a7fbe","#4aae8c","#e06c4a",
                  "#9b59b6","#e0a94a","#e05a7a"]
        has_map = any(r.map50 > 0 for r in self.results)

        panels = [
            ([r.map50      for r in self.results], "mAP50",     "mAP50 ↑"),
            ([r.map5095    for r in self.results], "mAP50-95",  "mAP50-95 ↑"),
            ([r.params/1e6 for r in self.results], "Params (M)","Params (M)"),
            ([r.gflops     for r in self.results], "GFLOPs",    "GFLOPs"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "YOLOv26n — Architectural Component Ablation Study",
            fontsize=14, fontweight="bold", y=0.98,
        )
        for ax, (values, ylabel, title) in zip(axes.flat, panels):
            bars = ax.bar(labels, values, color=colors[:len(values)],
                          width=0.55, edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    f"{val:.4f}" if val < 1 else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )
            if has_map and values[0] > 0:
                ax.axhline(values[0], color=colors[0], linestyle="--",
                           linewidth=0.8, alpha=0.5, label="A1 Full ref")
                ax.legend(fontsize=8)
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_ylim(0, max(values)*1.18 if max(values) > 0 else 1)
            ax.tick_params(axis="x", labelsize=7)

        plt.tight_layout(rect=[0,0,1,0.97])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Plot saved → {save_path}")
        plt.close()

    def plot_component_contribution(
        self, save_path: Optional[str] = None
    ) -> None:
        """Horizontal bar chart: mAP50 drop per removed component."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not any(r.map50 > 0 for r in self.results):
            log.info("Skipping contribution plot — no mAP data yet.")
            return

        save_path = save_path or os.path.join(
            self.cfg.project_dir, "component_contribution.png"
        )
        res_map = {r.variant: r for r in self.results}
        contrib = {
            "Attention (CBAM)"  : ("A1_Full","A2_NoAttention"),
            "PAN pathway"       : ("A1_Full","A3_NoPAN"),
            "FPN pathway"       : ("A1_Full","A4_NoFPN"),
            "Decoupled Head"    : ("A1_Full","A5_CoupledHead"),
            "Attention + PAN"   : ("A1_Full","A6_NoAttnNoPAN"),
        }
        labels, drops = [], []
        for comp, (ref_k, abl_k) in contrib.items():
            if ref_k in res_map and abl_k in res_map:
                labels.append(comp)
                drops.append(res_map[ref_k].map50 - res_map[abl_k].map50)

        if not drops:
            return

        colors = ["#e06c4a" if d > 0 else "#4aae8c" for d in drops]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(labels, drops, color=colors,
                       edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, drops):
            ax.text(
                val + 0.001 if val >= 0 else val - 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9,
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("mAP50 drop when component removed", fontsize=11)
        ax.set_title("YOLOv26n Component Contribution (mAP50 Δ)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Contribution plot saved → {save_path}")
        plt.close()


# =============================================================================
#  SECTION 8 — Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv26n Architectural Ablation Study"
    )
    parser.add_argument("--weights", default=cfg.model_weights)
    parser.add_argument("--data",    default=cfg.data_yaml)
    parser.add_argument("--epochs",  default=100,  type=int)
    parser.add_argument("--batch",   default=32,   type=int)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--nc",      default=25,   type=int)
    parser.add_argument("--variant", default=None,
                        help="Single variant e.g. A2_NoAttention")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print params/GFLOPs without training")
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    config               = ArchConfig()
    config.model_weights = args.weights
    config.data_yaml     = args.data
    config.epochs        = args.epochs
    config.batch         = args.batch
    config.device        = args.device
    config.num_classes   = args.nc

    study = ArchAblationStudy(config)

    if args.variant:
        study.run_variant(args.variant, dry_run=args.dry_run)
    else:
        study.run_all(dry_run=args.dry_run)

    study.print_summary()
    study.save_csv()

    if args.plot:
        study.plot_results()
        study.plot_component_contribution()