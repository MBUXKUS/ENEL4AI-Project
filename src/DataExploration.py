from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import DATA_DIR, PROJECT_ROOT

DATASET_DIR = DATA_DIR / "FruitDataset" / "Dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "dataset"
CLASS_DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.md"
REPORT_PATH = REPORTS_DIR / "data_exploration_report.md"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

image_paths = list(DATASET_DIR.rglob("*.jpg")) + list(DATASET_DIR.rglob("*.jpeg"))

splits = Counter()
fruits = Counter()
conditions = Counter()
fruit_condition = Counter()
widths = []
heights = []

for path in image_paths:
    split, fruit, condition = path.relative_to(DATASET_DIR).parts[:3]
    splits[split] += 1
    fruits[fruit] += 1
    conditions[condition] += 1
    fruit_condition[(fruit, condition)] += 1

    with Image.open(path) as img:
        w, h = img.size
    widths.append(w)
    heights.append(h)

resolutions = Counter(zip(widths, heights))

# --- Figures ---

class_figures = []
pixel_figures = []


def save_figure(filename, title, figure_list):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()
    figure_list.append((filename, title))


plt.figure()
plt.bar(fruits.keys(), fruits.values())
plt.xlabel("Fruit")
plt.ylabel("Count")
save_figure("counts_per_fruit.png", "Image count per fruit", class_figures)

plt.figure()
plt.bar(conditions.keys(), conditions.values())
plt.xlabel("Condition")
plt.ylabel("Count")
save_figure("counts_per_condition.png", "Image count per condition", class_figures)

for condition in conditions:
    counts = [fruit_condition[(fruit, condition)] for fruit in fruits]
    plt.figure()
    plt.pie(counts, labels=list(fruits.keys()), autopct="%1.1f%%")
    save_figure(f"counts_per_fruit_{condition.lower()}.png", f"Fruit share - {condition}", class_figures)

plt.figure()
plt.hist(widths, bins=30, alpha=0.6, label="Width")
plt.hist(heights, bins=30, alpha=0.6, label="Height")
plt.xlabel("Pixels")
plt.ylabel("Count")
plt.legend()
save_figure("pixel_dimensions.png", "Image width and height distribution", pixel_figures)

class_lines = []
class_lines.append("# Class Distribution")
class_lines.append("")
class_lines.append(f"Total images: {len(image_paths)}")
class_lines.append("")

class_lines.append("## Images per split")
class_lines.append("")
for split, count in splits.most_common():
    class_lines.append(f"- {split}: {count}")
class_lines.append("")

class_lines.append("## Images per fruit")
class_lines.append("")
for fruit, count in fruits.most_common():
    class_lines.append(f"- {fruit}: {count}")
class_lines.append("")

class_lines.append("## Images per condition")
class_lines.append("")
for condition, count in conditions.most_common():
    class_lines.append(f"- {condition}: {count}")
class_lines.append("")

class_lines.append("## Images per fruit x condition")
class_lines.append("")
class_lines.append("| Fruit | Condition | Count |")
class_lines.append("|---|---|---|")
for (fruit, condition), count in sorted(fruit_condition.items()):
    class_lines.append(f"| {fruit} | {condition} | {count} |")
class_lines.append("")

class_lines.append("## Figures")
class_lines.append("")
for name, title in class_figures:
    class_lines.append(f"### {title}")
    class_lines.append("")
    class_lines.append(f"![{title}](figures/dataset/{name})")
    class_lines.append("")

CLASS_DISTRIBUTION_PATH.write_text("\n".join(class_lines), encoding="utf-8")
print(f"Saved class distribution report to {CLASS_DISTRIBUTION_PATH}")

lines = []
lines.append("# Data Exploration Report")
lines.append("")
lines.append(f"Total images: {len(image_paths)}")
lines.append("")
lines.append("Class distribution (counts per split/fruit/condition and their figures) has moved to [class_distribution.md](class_distribution.md).")
lines.append("")

lines.append("## Pixel dimensions")
lines.append("")
lines.append(f"- Width  - min: {min(widths)}, max: {max(widths)}, mean: {sum(widths) / len(widths):.1f}")
lines.append(f"- Height - min: {min(heights)}, max: {max(heights)}, mean: {sum(heights) / len(heights):.1f}")
lines.append("")
lines.append("Most common resolutions (width x height):")
lines.append("")
for (w, h), count in resolutions.most_common(5):
    lines.append(f"- {w}x{h}: {count}")
lines.append("")

lines.append("## Figures")
lines.append("")
for name, title in pixel_figures:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(f"![{title}](figures/dataset/{name})")
    lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved data exploration report to {REPORT_PATH}")
