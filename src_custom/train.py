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
from model import VGGLiteGAP

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
    # condition (Fresh/Rotten/Formalin-mixed) is partly distinguished by color/sheen,
    # so keep color augmentation minimal to avoid training away that exact signal.
    color_jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            color_jitter,
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


def run_epoch(model, loader, criterion, optimizer=None, mixup_alpha=0.0, aux_weight=None):
    is_train = optimizer is not None
    use_aux = aux_weight is not None
    model.train(is_train)

    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    correct = 0
    aux_correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_actual = Counter()
    per_class_predicted = Counter()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if use_aux:
                images, labels, fruit_labels = batch
                fruit_labels = fruit_labels.to(DEVICE)
            else:
                images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train and mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
                outputs = model(images)
                logits = outputs[0] if use_aux else outputs
                main_loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                outputs = model(images)
                logits = outputs[0] if use_aux else outputs
                main_loss = criterion(logits, labels)

            if use_aux:
                aux_logits = outputs[1]
                aux_loss = criterion(aux_logits, fruit_labels)
                loss = main_loss + aux_weight * aux_loss
            else:
                loss = main_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_main_loss += main_loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if use_aux:
                total_aux_loss += aux_loss.item() * images.size(0)
                aux_correct += (aux_logits.argmax(dim=1) == fruit_labels).sum().item()

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                per_class_actual[label] += 1
                per_class_predicted[prediction] += 1
                if label == prediction:
                    per_class_correct[label] += 1

    metrics = {
        "loss": total_loss / total,
        "main_loss": total_main_loss / total,
        "accuracy": correct / total,
        "per_class_correct": per_class_correct,
        "per_class_actual": per_class_actual,
        "per_class_predicted": per_class_predicted,
    }
    if use_aux:
        metrics["aux_loss"] = total_aux_loss / total
        metrics["aux_accuracy"] = aux_correct / total
    return metrics


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--use-aux-head", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_transform, eval_transform = build_transforms(args.image_size)

    train_set = FruitDataset("train", transform=train_transform, return_fruit_label=args.use_aux_head)
    valid_set = FruitDataset("valid", transform=eval_transform, return_fruit_label=args.use_aux_head)
    test_set = FruitDataset("test", transform=eval_transform, return_fruit_label=args.use_aux_head)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classes = train_set.classes
    aux_classes = len(train_set.fruit_classes) if args.use_aux_head else None
    aux_weight = args.aux_weight if args.use_aux_head else None
    model = VGGLiteGAP(num_classes=len(classes), dropout=args.dropout, aux_classes=aux_classes).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())

    run_name = f"condition{'_aux' if args.use_aux_head else ''}"
    logger = Logger(LOG_DIR / f"custom_{run_name}_train_log.txt")
    logger.log(f"Classes: {classes}")
    logger.log(f"Model params: {num_params:,}")
    logger.log(f"Args: {vars(args)}")
    logger.log()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    checkpoint_path = CHECKPOINT_DIR / f"custom_{run_name}_best.pt"
    best_valid_accuracy = 0.0
    best_valid_metrics = None
    best_epoch = None
    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": [], "lr": []}

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, mixup_alpha=args.mixup_alpha, aux_weight=aux_weight)
        valid_metrics = run_epoch(model, valid_loader, criterion, aux_weight=aux_weight)
        scheduler.step(valid_metrics["accuracy"])

        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["valid_acc"].append(valid_metrics["accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        log_line = (
            f"[{epoch + 1}/{args.epochs}] "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.1%} | "
            f"valid loss {valid_metrics['loss']:.4f} acc {valid_metrics['accuracy']:.1%} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if args.use_aux_head:
            log_line += (
                f" | train aux loss {train_metrics['aux_loss']:.4f} acc {train_metrics['aux_accuracy']:.1%}"
                f" | valid aux loss {valid_metrics['aux_loss']:.4f} acc {valid_metrics['aux_accuracy']:.1%}"
            )
        logger.log(log_line)

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            best_valid_metrics = valid_metrics
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_ax.plot(epochs_range, history["train_loss"], label="train")
    loss_ax.plot(epochs_range, history["valid_loss"], label="valid")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss")
    loss_ax.legend()

    acc_ax.plot(epochs_range, history["train_acc"], label="train")
    acc_ax.plot(epochs_range, history["valid_acc"], label="valid")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()

    fig.suptitle(f"Training curves ({run_name})")
    fig.tight_layout()
    curves_path = FIGURES_DIR / f"training_curves_{run_name}.png"
    fig.savefig(curves_path)
    plt.close(fig)
    logger.log(f"\nSaved training curves to {curves_path}")

    logger.log(f"\nBest valid accuracy: {best_valid_accuracy:.1%} (epoch {best_epoch}/{args.epochs})")
    logger.log(f"Per-class valid metrics (best epoch, {best_epoch}):")
    print_per_class_metrics(
        classes, best_valid_metrics["per_class_correct"], best_valid_metrics["per_class_actual"], best_valid_metrics["per_class_predicted"],
        log_fn=logger.log,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_metrics = run_epoch(model, test_loader, criterion, aux_weight=aux_weight)
    logger.log(f"\nTest accuracy: {test_metrics['accuracy']:.1%}")
    logger.log("Per-class test metrics:")
    print_per_class_metrics(
        classes, test_metrics["per_class_correct"], test_metrics["per_class_actual"], test_metrics["per_class_predicted"],
        log_fn=logger.log,
    )
    logger.close()


if __name__ == "__main__":
    main()
