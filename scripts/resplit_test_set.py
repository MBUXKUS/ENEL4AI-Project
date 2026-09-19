"""Fixes data/FruitDataset/Dataset/test properly.

First attempt (see git history) moved a random 15% sample of each train
bucket into test. That caused severe leakage: filenames encode near-
sequential capture IDs (burst/video frames of the same physical fruit,
e.g. 6102784868896325430, 6102784868896325431, ...), and random sampling
interleaved train/test throughout that sequence -- every test image ended
up with near-duplicate siblings still sitting in train (confirmed: 100% of
test IDs in Apple/Rotten fell inside train's ID range). This is why test
accuracy was implausibly higher than validation accuracy.

Fix: merge test back into train, then for each (fruit, condition) bucket,
sort all files by their numeric ID and cut out a *contiguous* block near
the target test fraction, choosing the cut point at the largest ID gap in
that region (a likely session/burst boundary) rather than an arbitrary
index. Files with no numeric ID (a small "add (N).jpg" batch) are left in
train, since they have no sequence info to place safely.
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src_custom"))
from config import DATA_DIR

SEED = 42
TEST_FRACTION = 0.15
SEARCH_WINDOW_FRACTION = 0.5  # search +/- this fraction of the target count around the target cut index

ID_PATTERN = re.compile(r"(\d+)\.jpe?g$", re.IGNORECASE)

DATASET_DIR = DATA_DIR / "FruitDataset" / "Dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

random.seed(SEED)


def extract_id(path):
    match = ID_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


total_moved_back = 0
total_test = 0
for fruit_dir in sorted(TRAIN_DIR.iterdir()):
    if not fruit_dir.is_dir():
        continue
    for condition_dir in sorted(fruit_dir.iterdir()):
        if not condition_dir.is_dir():
            continue

        test_condition_dir = TEST_DIR / fruit_dir.name / condition_dir.name

        # Step 1: merge anything currently in test back into train.
        for path in list(test_condition_dir.glob("*.jpg")) + list(test_condition_dir.glob("*.jpeg")):
            path.rename(condition_dir / path.name)
            total_moved_back += 1

        # Step 2: split by sorted numeric ID, cutting at the largest gap near the target fraction.
        all_files = list(condition_dir.glob("*.jpg")) + list(condition_dir.glob("*.jpeg"))
        with_id = sorted((p for p in all_files if extract_id(p) is not None), key=extract_id)
        without_id = [p for p in all_files if extract_id(p) is None]

        n = len(with_id)
        target_cut = n - round(n * TEST_FRACTION)  # everything from this index onward -> test
        window = max(1, round(n * TEST_FRACTION * SEARCH_WINDOW_FRACTION))
        low = max(1, target_cut - window)
        high = min(n - 1, target_cut + window)

        ids = [extract_id(p) for p in with_id]
        best_cut = target_cut
        best_gap = -1
        for i in range(low, high + 1):
            gap = ids[i] - ids[i - 1]
            if gap > best_gap:
                best_gap = gap
                best_cut = i

        test_files = with_id[best_cut:]

        test_condition_dir.mkdir(parents=True, exist_ok=True)
        for path in test_files:
            path.rename(test_condition_dir / path.name)

        total_test += len(test_files)
        print(f"{fruit_dir.name}/{condition_dir.name}: {len(test_files)} to test (of {n} numbered, {len(without_id)} unnumbered left in train), cut gap={best_gap}")

print(f"\nMoved back from old (leaky) test: {total_moved_back}")
print(f"Total in new test: {total_test}")
