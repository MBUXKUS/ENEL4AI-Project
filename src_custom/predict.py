import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE
from dataset import FruitDataset
from model import VGGLiteGAP

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_DIR = DATA_DIR / "FruitDataset" / "Dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "custom_condition_best.pt"))
    parser.add_argument("--image", default=None, help="Path to an image; if omitted, picks a random one from the test split")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    classes = FruitDataset("test").classes

    if args.image is None:
        test_dir = DATASET_DIR / "test"
        candidates = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg"))
        image_path = random.choice(candidates)
        true_label = image_path.parent.name
    else:
        image_path = Path(args.image)
        true_label = None

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = VGGLiteGAP(num_classes=len(classes)).to(DEVICE)
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)[0]

    predicted_idx = probs.argmax().item()

    print(f"Image: {image_path}")
    if true_label is not None:
        print(f"True label: {true_label}")
    print(f"Predicted: {classes[predicted_idx]} ({probs[predicted_idx]:.1%} confidence)")
    print("\nAll class probabilities:")
    for idx, name in enumerate(classes):
        print(f"  {name:20s} {probs[idx]:.1%}")


if __name__ == "__main__":
    main()
