"""Generates block diagrams of both model architectures for the report, based on the
actual layer sequences in src_custom/model.py and src/model.py (EfficientNetV2-S
stage structure verified against the real torchvision module, not from memory)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src_custom"))
from config import FIGURES_DIR

COLORS = {
    "io": "#4a4a4a",
    "conv": "#4C72B0",
    "norm": "#8CA9D6",
    "act": "#DD8452",
    "pool": "#55A868",
    "se": "#8172B2",
    "merge": "#CCB974",
    "dense": "#64B5CD",
    "reg": "#B0B0B0",
    "block": "#4C72B0",
    "backbone": "#9AB6DE",
    "new_head": "#DD8452",
}


def draw_diagram(boxes, title, save_path, box_width=6.4, box_height=0.55, gap=0.16, fig_width=8.5):
    n = len(boxes)
    fig_height = n * (box_height + gap) + 1.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y = n * (box_height + gap)
    centers = []
    for box in boxes:
        y -= box_height + gap
        centers.append(y + box_height / 2)
        color = COLORS.get(box.get("color", "reg"), "#B0B0B0")
        rect = FancyBboxPatch(
            (0, y), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1, edgecolor="black", facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)
        label = box["label"]
        shape = box.get("shape", "")
        text = f"{label}\n{shape}" if shape else label
        ax.text(box_width / 2, y + box_height / 2, text, ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold", linespacing=1.3)

    for i in range(n - 1):
        y0 = centers[i] - box_height / 2
        y1 = centers[i + 1] + box_height / 2
        ax.add_patch(FancyArrow(box_width / 2, y0 - 0.01, 0, (y1 - y0) + 0.02,
                                 width=0.008, head_width=0.12, head_length=0.06,
                                 length_includes_head=True, color="black"))

    ax.set_xlim(-0.3, box_width + 0.3)
    ax.set_ylim(-0.3, n * (box_height + gap) + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def custom_cnn_diagram():
    boxes = [
        {"label": "Input", "shape": "3 x 224 x 224", "color": "io"},

        {"label": "Block 1 -- Conv3x3 (3->32), no bias", "color": "conv"},
        {"label": "BatchNorm2d(32) -> ReLU", "color": "norm"},
        {"label": "Conv3x3 (32->32), no bias -> BatchNorm2d(32)", "color": "conv"},
        {"label": "Shortcut: Conv1x1 (3->32) -> BatchNorm2d(32)", "color": "merge"},
        {"label": "Squeeze-and-Excitation (reduction=8)", "color": "se"},
        {"label": "Add (main + shortcut) -> ReLU -> MaxPool2x2", "shape": "32 x 112 x 112", "color": "pool"},

        {"label": "Block 2 -- ResConvBlock (32->64)\n(same internal layers as Block 1)", "shape": "64 x 56 x 56", "color": "block"},
        {"label": "Block 3 -- ResConvBlock (64->128)", "shape": "128 x 28 x 28", "color": "block"},
        {"label": "Block 4 -- ResConvBlock (128->256)", "shape": "256 x 14 x 14", "color": "block"},
        {"label": "Block 5 -- ResConvBlock (256->512)", "shape": "512 x 7 x 7", "color": "block"},

        {"label": "GlobalAvgPool2d  ||  GlobalMaxPool2d", "shape": "512 + 512", "color": "pool"},
        {"label": "Concatenate (GAP, GMP)", "shape": "1024", "color": "merge"},
        {"label": "Dropout(p=0.5)", "color": "reg"},
        {"label": "Linear(1024 -> 256)", "color": "dense"},
        {"label": "ReLU", "color": "act"},
        {"label": "Dropout(p=0.5)", "color": "reg"},
        {"label": "Linear(256 -> num_classes)", "color": "dense"},
        {"label": "Output logits", "shape": "3 (condition) or 15 (fruit x condition)", "color": "io"},
    ]
    draw_diagram(boxes, "Custom CNN -- VGGLiteGAP (~5.24M params, trained from scratch)",
                 FIGURES_DIR / "architecture_custom_cnn.png")


def transfer_learning_diagram():
    boxes = [
        {"label": "Input", "shape": "3 x 260 x 260", "color": "io"},

        {"label": "Stem: Conv3x3, stride 2 (3->24) + BN + SiLU", "shape": "24 x 130 x 130", "color": "backbone"},
        {"label": "Stage 1: Fused-MBConv1 x2 (24->24)", "shape": "24 x 130 x 130", "color": "backbone"},
        {"label": "Stage 2: Fused-MBConv4 x4 (24->48), stride 2", "shape": "48 x 65 x 65", "color": "backbone"},
        {"label": "Stage 3: Fused-MBConv4 x4 (48->64), stride 2", "shape": "64 x 33 x 33", "color": "backbone"},
        {"label": "Stage 4: MBConv4 + SE x6 (64->128), stride 2", "shape": "128 x 17 x 17", "color": "backbone"},
        {"label": "Stage 5: MBConv6 + SE x9 (128->160)", "shape": "160 x 17 x 17", "color": "backbone"},
        {"label": "Stage 6: MBConv6 + SE x15 (160->256), stride 2", "shape": "256 x 9 x 9", "color": "backbone"},
        {"label": "Head: Conv1x1 (256->1280) + BN + SiLU", "shape": "1280 x 9 x 9", "color": "backbone"},

        {"label": "AdaptiveAvgPool2d", "shape": "1280", "color": "pool"},
        {"label": "Dropout(p=0.2)", "color": "reg"},
        {"label": "Linear(1280 -> num_classes)\n[replaces original 1000-class ImageNet head]", "color": "new_head"},
        {"label": "Output logits", "shape": "3 (condition)", "color": "io"},
    ]
    draw_diagram(boxes, "Transfer Learning -- EfficientNetV2-S (~20.18M params, ImageNet-pretrained)\nBlue = pretrained backbone (3-phase progressive unfreeze) | Orange = newly trained head",
                 FIGURES_DIR / "architecture_transfer_learning.png", fig_width=9.0)


if __name__ == "__main__":
    custom_cnn_diagram()
    transfer_learning_diagram()
