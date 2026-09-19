import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DEVICE, FIGURES_DIR, REPORTS_DIR
from dataset import FruitDataset
from model import build_model
from train import build_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_v2_s", choices=["resnet50", "efficientnet_v2_s", "efficientnet_v2_m", "convnext_tiny"])
    parser.add_argument("--checkpoint", default=None, help="Defaults to outputs/checkpoints/{model}_condition_best.pt")
    parser.add_argument("--image-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation (flip-averaging)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or str(CHECKPOINT_DIR / f"{args.model}_condition_best.pt")
    use_tta = not args.no_tta

    _, eval_transform = build_transforms(args.image_size)
    test_set = FruitDataset("test", transform=eval_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    classes = test_set.classes
    num_classes = len(classes)

    model, _ = build_model(args.model, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    confusion = np.zeros((num_classes, num_classes), dtype=int)  # rows: true, cols: predicted
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1)
            if use_tta:
                flipped_probs = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
                probs = (probs + flipped_probs) / 2
            probs = probs.cpu()
            predictions = probs.argmax(dim=1)
            for true_idx, pred_idx in zip(labels.tolist(), predictions.tolist()):
                confusion[true_idx, pred_idx] += 1
            all_probs.append(probs.numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # [N, num_classes]
    all_labels = np.concatenate(all_labels, axis=0)  # [N]

    total = confusion.sum()
    accuracy = np.trace(confusion) / total

    per_class = []
    for idx, name in enumerate(classes):
        tp = confusion[idx, idx]
        actual = confusion[idx, :].sum()
        predicted = confusion[:, idx].sum()
        recall = tp / actual if actual else float("nan")
        precision = tp / predicted if predicted else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        per_class.append((name, recall, precision, f1, actual))

    macro_recall = np.nanmean([row[1] for row in per_class])
    macro_precision = np.nanmean([row[2] for row in per_class])
    macro_f1 = np.nanmean([row[3] for row in per_class])

    tta_label = "with TTA" if use_tta else "no TTA"
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test accuracy ({tta_label}): {accuracy:.1%}")
    print(f"Macro precision: {macro_precision:.1%} | macro recall: {macro_recall:.1%} | macro F1: {macro_f1:.1%}\n")
    print(f"  {'class':30s} {'recall':>8s} {'precision':>10s} {'f1':>8s} {'support':>8s}")
    for name, recall, precision, f1, actual in per_class:
        print(f"  {name:30s} {recall:>8.1%} {precision:>10.1%} {f1:>8.1%} {actual:>8d}")

    suffix = f"{args.model}_condition"

    # Per-class precision/recall/F1 bar chart.
    x = np.arange(num_classes)
    width = 0.25
    plt.figure(figsize=(1.8 * num_classes + 2, 5))
    plt.bar(x - width, [row[2] for row in per_class], width, label="Precision")
    plt.bar(x, [row[1] for row in per_class], width, label="Recall")
    plt.bar(x + width, [row[3] for row in per_class], width, label="F1")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Per-class precision / recall / F1 ({args.model}, {tta_label})")
    plt.legend()
    plt.tight_layout()
    metrics_figure_name = f"per_class_metrics_{suffix}.png"
    plt.savefig(FIGURES_DIR / metrics_figure_name)
    plt.close()

    # Confusion matrix figure.
    plt.figure(figsize=(1.5 * num_classes + 2, 1.5 * num_classes + 2))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(num_classes), classes, rotation=45, ha="right")
    plt.yticks(range(num_classes), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix ({args.model}, {tta_label})")
    threshold = confusion.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if confusion[i, j] > threshold else "black"
            plt.text(j, i, confusion[i, j], ha="center", va="center", color=color)
    plt.tight_layout()
    figure_name = f"confusion_matrix_{suffix}.png"
    plt.savefig(FIGURES_DIR / figure_name)
    plt.close()

    # ROC curves (one-vs-rest per class) with AUC, computed manually (no sklearn dependency).
    def roc_curve(y_true_binary, scores):
        order = np.argsort(-scores)
        y_sorted = y_true_binary[order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        n_pos = tps[-1] if len(tps) else 0
        n_neg = fps[-1] if len(fps) else 0
        tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)
        fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps, dtype=float)
        tpr = np.concatenate(([0.0], tpr, [1.0]))
        fpr = np.concatenate(([0.0], fpr, [1.0]))
        auc = np.trapezoid(tpr, fpr)
        return fpr, tpr, auc

    plt.figure(figsize=(6, 6))
    for idx, name in enumerate(classes):
        y_true_binary = (all_labels == idx).astype(int)
        fpr, tpr, auc = roc_curve(y_true_binary, all_probs[:, idx])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves, one-vs-rest ({args.model}, {tta_label})")
    plt.legend()
    plt.tight_layout()
    roc_figure_name = f"roc_curve_{suffix}.png"
    plt.savefig(FIGURES_DIR / roc_figure_name)
    plt.close()

    # Markdown report.
    lines = [
        f"# Evaluation Report ({args.model}, condition)",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Test-time augmentation: {tta_label}",
        "",
        f"**Test accuracy: {accuracy:.1%}**",
        "",
        f"Macro precision: {macro_precision:.1%} | Macro recall: {macro_recall:.1%} | Macro F1: {macro_f1:.1%}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Recall | Precision | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name, recall, precision, f1, actual in per_class:
        lines.append(f"| {name} | {recall:.1%} | {precision:.1%} | {f1:.1%} | {actual} |")
    lines += [
        "",
        "## Per-class precision / recall / F1",
        "",
        f"![Per-class metrics](figures/{metrics_figure_name})",
        "",
        "## Confusion matrix",
        "",
        f"![Confusion matrix](figures/{figure_name})",
        "",
        "## ROC curves (one-vs-rest)",
        "",
        f"![ROC curves](figures/{roc_figure_name})",
        "",
    ]
    report_path = REPORTS_DIR / f"evaluation_report_{suffix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report to {report_path}")
    print(f"Saved per-class metrics chart to {FIGURES_DIR / metrics_figure_name}")
    print(f"Saved confusion matrix to {FIGURES_DIR / figure_name}")
    print(f"Saved ROC curves to {FIGURES_DIR / roc_figure_name}")


if __name__ == "__main__":
    main()
