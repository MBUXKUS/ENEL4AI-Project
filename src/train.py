import argparse
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import CHECKPOINT_DIR, DEVICE, FIGURES_DIR, LOG_DIR
from dataset import FruitDataset
from model import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Logger:
    """Tees print()-style output to both the console and a log file on disk."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")

    def log(self, message=""):
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # Kept mild and grayscale-only: condition (Fresh/Rotten/Formalin-mixed) is
            # partly distinguished by color/sheen, so avoid training that signal away.
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def mixup_batch(images, labels, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def run_epoch(model, loader, criterion, optimizer=None, mixup_alpha=0.0):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_actual = Counter()
    per_class_predicted = Counter()

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train and mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                per_class_actual[label] += 1
                per_class_predicted[prediction] += 1
                if label == prediction:
                    per_class_correct[label] += 1

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class_correct": per_class_correct,
        "per_class_actual": per_class_actual,
        "per_class_predicted": per_class_predicted,
    }


def evaluate_with_tta(model, loader):
    """Averages predictions over the original image and its horizontal flip.
    Free accuracy at inference time, no retraining needed."""
    model.eval()

    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_actual = Counter()
    per_class_predicted = Counter()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            probs = torch.softmax(model(images), dim=1)
            flipped_probs = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            predictions = ((probs + flipped_probs) / 2).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                per_class_actual[label] += 1
                per_class_predicted[prediction] += 1
                if label == prediction:
                    per_class_correct[label] += 1

    return {
        "accuracy": correct / total,
        "per_class_correct": per_class_correct,
        "per_class_actual": per_class_actual,
        "per_class_predicted": per_class_predicted,
    }


def print_per_class_metrics(classes, per_class_correct, per_class_actual, per_class_predicted, log_fn=print):
    log_fn(f"  {'class':30s} {'recall':>8s} {'precision':>10s} {'f1':>8s}")
    for idx, name in enumerate(classes):
        actual = per_class_actual.get(idx, 0)
        predicted = per_class_predicted.get(idx, 0)
        correct = per_class_correct.get(idx, 0)
        recall = correct / actual if actual else float("nan")
        precision = correct / predicted if predicted else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        log_fn(f"  {name:30s} {recall:>8.1%} {precision:>10.1%} {f1:>8.1%}")


def run_phase(model, train_loader, valid_loader, criterion, optimizer, epochs, phase_name, mixup_alpha, checkpoint_path, state, history, log_fn, scheduler=None):
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, mixup_alpha=mixup_alpha)
        valid_metrics = run_epoch(model, valid_loader, criterion)
        if scheduler is not None:
            scheduler.step(valid_metrics["accuracy"])

        state["global_epoch"] += 1
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(state["global_epoch"])
        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["valid_acc"].append(valid_metrics["accuracy"])

        log_fn(
            f"[{phase_name} {epoch + 1}/{epochs}] "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.1%} | "
            f"valid loss {valid_metrics['loss']:.4f} acc {valid_metrics['accuracy']:.1%} | "
            f"lr {lr:.2e}"
        )

        if valid_metrics["accuracy"] > state["best_accuracy"]:
            state["best_accuracy"] = valid_metrics["accuracy"]
            state["best_metrics"] = valid_metrics
            state["best_epoch"] = state["global_epoch"]
            state["best_phase"] = phase_name
            torch.save(model.state_dict(), checkpoint_path)

    history["phase_boundaries"].append((phase_name, state["global_epoch"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_v2_s", choices=["resnet50", "efficientnet_v2_s", "efficientnet_v2_m", "convnext_tiny"])
    parser.add_argument("--image-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--partial-unfreeze-epochs", type=int, default=5)
    parser.add_argument("--partial-unfreeze-fraction", type=float, default=0.3)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--partial-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_transform, eval_transform = build_transforms(args.image_size)

    train_set = FruitDataset("train", transform=train_transform)
    valid_set = FruitDataset("valid", transform=eval_transform)
    test_set = FruitDataset("test", transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classes = train_set.classes
    model, backbone_params = build_model(args.model, num_classes=len(classes))
    model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())

    logger = Logger(LOG_DIR / f"{args.model}_condition_train_log.txt")
    logger.log(f"Model: {args.model} | classes: {classes}")
    logger.log(f"Model params: {num_params:,}")
    logger.log(f"Args: {vars(args)}")
    logger.log()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_condition_best.pt"

    state = {"global_epoch": 0, "best_accuracy": 0.0, "best_metrics": None, "best_epoch": None, "best_phase": None}
    history = {"epoch": [], "train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": [], "phase_boundaries": []}

    # Phase 1: freeze backbone, train the new head only.
    for param in backbone_params:
        param.requires_grad = False
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.head_lr)
    run_phase(model, train_loader, valid_loader, criterion, optimizer, args.freeze_epochs, "head", args.mixup_alpha, checkpoint_path, state, history, logger.log)

    # Phase 2: unfreeze the deeper fraction of the backbone (later layers, more task-specific),
    # keep earlier layers frozen, train at an intermediate LR. Avoids disturbing early pretrained
    # features while adapting the more specialized late features to this task.
    n_partial = int(len(backbone_params) * args.partial_unfreeze_fraction)
    for param in backbone_params[-n_partial:]:
        param.requires_grad = True
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.partial_lr)
    run_phase(model, train_loader, valid_loader, criterion, optimizer, args.partial_unfreeze_epochs, "partial", args.mixup_alpha, checkpoint_path, state, history, logger.log)

    # Phase 3: unfreeze everything, fine-tune at the lowest LR with plateau-based decay.
    for param in backbone_params:
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    run_phase(model, train_loader, valid_loader, criterion, optimizer, args.finetune_epochs, "finetune", args.mixup_alpha, checkpoint_path, state, history, logger.log, scheduler=scheduler)

    # Training curves, with dashed lines marking phase transitions.
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_ax.plot(history["epoch"], history["train_loss"], label="train")
    loss_ax.plot(history["epoch"], history["valid_loss"], label="valid")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss")

    acc_ax.plot(history["epoch"], history["train_acc"], label="train")
    acc_ax.plot(history["epoch"], history["valid_acc"], label="valid")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy")

    for phase_name, boundary_epoch in history["phase_boundaries"][:-1]:
        loss_ax.axvline(boundary_epoch + 0.5, color="gray", linestyle="--", linewidth=1)
        acc_ax.axvline(boundary_epoch + 0.5, color="gray", linestyle="--", linewidth=1)
    loss_ax.legend()
    acc_ax.legend()

    fig.suptitle(f"Training curves ({args.model}, condition) — dashed lines mark phase transitions")
    fig.tight_layout()
    curves_path = FIGURES_DIR / f"training_curves_{args.model}_condition.png"
    fig.savefig(curves_path)
    plt.close(fig)
    logger.log(f"\nSaved training curves to {curves_path}")

    logger.log(f"\nBest valid accuracy: {state['best_accuracy']:.1%} (epoch {state['best_epoch']}, {state['best_phase']} phase)")
    logger.log(f"Per-class valid metrics (best epoch, {state['best_epoch']}):")
    print_per_class_metrics(
        classes, state["best_metrics"]["per_class_correct"], state["best_metrics"]["per_class_actual"], state["best_metrics"]["per_class_predicted"],
        log_fn=logger.log,
    )

    # Final test evaluation using the best checkpoint: plain single-pass and TTA (flip-averaged).
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    test_metrics = run_epoch(model, test_loader, criterion)
    logger.log(f"\nTest accuracy (no TTA): {test_metrics['accuracy']:.1%}")
    logger.log("Per-class test metrics (no TTA):")
    print_per_class_metrics(
        classes, test_metrics["per_class_correct"], test_metrics["per_class_actual"], test_metrics["per_class_predicted"],
        log_fn=logger.log,
    )

    tta_metrics = evaluate_with_tta(model, test_loader)
    logger.log(f"\nTest accuracy (with TTA): {tta_metrics['accuracy']:.1%}")
    logger.log("Per-class test metrics (with TTA):")
    print_per_class_metrics(
        classes, tta_metrics["per_class_correct"], tta_metrics["per_class_actual"], tta_metrics["per_class_predicted"],
        log_fn=logger.log,
    )
    logger.close()


if __name__ == "__main__":
    main()
