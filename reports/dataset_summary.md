# Dataset Summary

## Source

- **Dataset:** Fruits Disease Dataset
- **Author:** saravanansri (Kaggle)
- **URL:** https://www.kaggle.com/datasets/saravanansri/fruits-disease-dataset/data
- **License / terms:** See the Kaggle dataset page for the license and usage terms before redistributing or publishing results based on this data.
- **Local path:** `data/FruitDataset/Dataset/{train,valid,test}/<Fruit>/<Condition>/*.jpg`

The dataset ships pre-split into `train`, `valid`, and `test` folders, each containing one subfolder per fruit, each in turn split into one subfolder per condition label.

## Size

| Split | Images | Share |
|---|---|---|
| train | 7,107 | 69.9% |
| valid | 1,528 | 15.0% |
| test  | 1,528 | 15.0% |
| **Total** | **10,163** | 100% |

## Classes

The data is organized along two independent label axes:

- **Fruit (5 classes):** Apple, Banana, Grape, Mango, Orange
- **Condition (3 classes):** Fresh, Formalin-mixed, Rotten

Combined, this gives 15 fine-grained (fruit × condition) classes. Depending on the modeling approach, these can be used directly (15-way classification), collapsed to just the condition axis (3-way: Fresh / Formalin-mixed / Rotten), or further collapsed to a binary defect / no-defect label (Fresh = no-defect, {Formalin-mixed, Rotten} = defect).

**Note on label semantics:** "Formalin-mixed" refers to chemical adulteration (fruit treated with formalin to extend shelf life), which is a food-safety/contamination label rather than a visual bruise/rot severity grade. This differs from the "defect vs. no-defect + severity grading" framing in the project brief — the closest available severity axis here is Fresh → Formalin-mixed → Rotten if treated as an ordinal condition scale, but Formalin-mixed is not necessarily visually intermediate between Fresh and Rotten. This should be made an explicit modeling decision rather than assumed.

## Class balance

**By fruit** (well balanced, ~2% spread across classes):

| Fruit | Count |
|---|---|
| Orange | 2,053 |
| Banana | 2,044 |
| Apple | 2,041 |
| Grape | 2,013 |
| Mango | 2,012 |

**By condition** (Fresh is ~19–20% larger than the other two):

| Condition | Count |
|---|---|
| Fresh | 3,804 |
| Formalin-mixed | 3,181 |
| Rotten | 3,178 |

**By fruit × condition:**

| Fruit | Condition | Count |
|---|---|---|
| Apple | Formalin-mixed | 644 |
| Apple | Fresh | 766 |
| Apple | Rotten | 631 |
| Banana | Formalin-mixed | 662 |
| Banana | Fresh | 750 |
| Banana | Rotten | 632 |
| Grape | Formalin-mixed | 611 |
| Grape | Fresh | 771 |
| Grape | Rotten | 631 |
| Mango | Formalin-mixed | 617 |
| Mango | Fresh | 764 |
| Mango | Rotten | 631 |
| Orange | Formalin-mixed | 647 |
| Orange | Fresh | 753 |
| Orange | Rotten | 653 |

Overall the dataset is close to balanced on both axes; the main imbalance to account for is Fresh (~37% of images) vs. Formalin-mixed/Rotten (~31% each) if training a condition-only classifier — mild enough that class weighting or oversampling is optional rather than mandatory, but worth monitoring per-class recall regardless.

## Image properties

- Format: JPEG
- Width: min 633px, max 4160px, mean 1097.5px
- Height: min 720px, max 4472px, mean 1257.2px
- Dominant resolutions: 960×1280 (5,673 images), 1280×960 (2,512 images), plus a long tail of other resolutions (780×1040, 721×1280, 1600×1200, etc.)

Images are portrait- and landscape-oriented in a roughly 3:4 / 4:3 ratio, at high resolution relative to typical CNN input sizes, so a resize/downsample step is required before training.

![Image width and height distribution](figures/pixel_dimensions.png)

## Preprocessing steps

**Completed:**
- Enumerated all images across splits/fruits/conditions and validated they open correctly via PIL (`src/DataExploration.py`).
- Computed and visualized class counts and pixel-dimension statistics (see figures below).

**Not yet implemented — planned:**
1. **Resize/crop** to a fixed square input size appropriate for the chosen backbone (e.g., 224×224 for MobileNetV2/ResNet, or the model's native input size for EfficientNet/etc.), preserving aspect ratio via center-crop or padding rather than naive stretch.
2. **Normalization** of pixel values to the mean/std expected by the pretrained backbone (or dataset-computed mean/std if training from scratch).
3. **Augmentation (train split only):** horizontal flip, small rotations, and mild brightness/contrast jitter to improve generalization. Aggressive hue/saturation jitter should be avoided or used sparingly, since color is part of the discriminative signal between Fresh, Formalin-mixed, and Rotten — over-augmenting color could erase the label signal itself.
4. **Corrupt/duplicate image check:** run a file-integrity pass and a perceptual-hash/duplicate scan, since Kaggle produce-quality datasets commonly contain near-duplicate or augmented copies that can leak across train/valid/test if not filtered.
5. **Split-leakage check:** confirm no identical or near-identical images (e.g., same fruit photographed from consecutive frames) appear in more than one of train/valid/test, which would inflate validation/test metrics.
6. **Label-mapping decision:** finalize whether the model targets are 15-way (fruit × condition), 3-way (condition only), or binary (defect / no-defect), and encode that mapping in a single reusable module rather than ad hoc per script.

## Figures

### Image count per fruit
![Image count per fruit](figures/counts_per_fruit.png)

### Image count per condition
![Image count per condition](figures/counts_per_condition.png)

### Fruit share — Fresh
![Fruit share - Fresh](figures/counts_per_fruit_fresh.png)

### Fruit share — Formalin-mixed
![Fruit share - Formalin-mixed](figures/counts_per_fruit_formalin-mixed.png)

### Fruit share — Rotten
![Fruit share - Rotten](figures/counts_per_fruit_rotten.png)

## Open items before this dataset is training-ready

- [ ] Confirm license/attribution terms from the Kaggle page and record them here.
- [ ] Decide the label taxonomy (15-way / 3-way / binary) for the classifier.
- [ ] Implement corrupt-file and duplicate/near-duplicate detection.
- [ ] Verify no train/valid/test leakage.
- [ ] Implement the resize/normalize/augmentation pipeline as reusable code (e.g., `src/transforms.py`) rather than only exploratory scripts.
