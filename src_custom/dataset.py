from PIL import Image
from torch.utils.data import Dataset

from config import DATA_DIR

DATASET_DIR = DATA_DIR / "FruitDataset" / "Dataset"


class FruitDataset(Dataset):
    """3-class dataset: one class per condition (Fresh/Rotten/Formalin-mixed), fruit identity discarded."""

    def __init__(self, split, transform=None, return_fruit_label=False):
        self.split_dir = DATASET_DIR / split
        self.transform = transform
        self.return_fruit_label = return_fruit_label

        pairs = sorted(
            (fruit_dir.name, condition_dir.name)
            for fruit_dir in self.split_dir.iterdir()
            if fruit_dir.is_dir()
            for condition_dir in fruit_dir.iterdir()
            if condition_dir.is_dir()
        )

        self.classes = sorted({condition for _, condition in pairs})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.fruit_classes = sorted({fruit for fruit, _ in pairs})
        self.fruit_to_idx = {name: idx for idx, name in enumerate(self.fruit_classes)}

        self.samples = []
        for fruit, condition in pairs:
            class_idx = self.class_to_idx[condition]
            fruit_idx = self.fruit_to_idx[fruit]
            condition_dir = self.split_dir / fruit / condition
            for path in list(condition_dir.glob("*.jpg")) + list(condition_dir.glob("*.jpeg")):
                self.samples.append((path, class_idx, fruit_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_idx, fruit_idx = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_fruit_label:
            return image, class_idx, fruit_idx
        return image, class_idx
