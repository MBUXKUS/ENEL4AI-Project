"""Simple desktop GUI: pick an image, pick a model (custom CNN or transfer learning),
see the predicted condition, confidence, and true label (if the image comes from the
labeled test set). No CLI args needed -- just run it.

src/ and src_custom/ each have their own config.py/dataset.py/model.py using bare
imports (from config import ..., etc.), so both can't be on sys.path at once without
one shadowing the other. load_package() below loads one package's modules, grabs the
specific objects needed, then clears sys.modules/sys.path before the other package is
loaded -- avoiding that collision.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import torch
from PIL import Image, ImageTk
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["Formalin-mixed", "Fresh", "Rotten"]


def load_package(package_dir):
    package_dir = str(package_dir)
    for name in ("config", "dataset", "model", "train"):
        sys.modules.pop(name, None)
    sys.path.insert(0, package_dir)
    import config as config_mod
    import model as model_mod
    modules = {"config": config_mod, "model": model_mod}
    sys.path.remove(package_dir)
    return modules


custom_pkg = load_package(PROJECT_ROOT / "src_custom")
CustomVGGLiteGAP = custom_pkg["model"].VGGLiteGAP
CUSTOM_CHECKPOINT = custom_pkg["config"].CHECKPOINT_DIR / "custom_condition_best.pt"
CUSTOM_DEVICE = custom_pkg["config"].DEVICE
CUSTOM_IMAGE_SIZE = 224

pretrained_pkg = load_package(PROJECT_ROOT / "src")
PretrainedBuildModel = pretrained_pkg["model"].build_model
PRETRAINED_CHECKPOINT = pretrained_pkg["config"].CHECKPOINT_DIR / "efficientnet_v2_s_condition_best.pt"
PRETRAINED_DEVICE = pretrained_pkg["config"].DEVICE
PRETRAINED_IMAGE_SIZE = 260

DATASET_TEST_DIR = PROJECT_ROOT / "data" / "FruitDataset" / "Dataset" / "test"

MODEL_OPTIONS = {
    "Custom CNN (VGGLiteGAP)": "custom",
    "Transfer Learning (EfficientNetV2-S)": "pretrained",
}

_loaded_models = {}


def get_model(key):
    if key in _loaded_models:
        return _loaded_models[key]

    if key == "custom":
        model = CustomVGGLiteGAP(num_classes=len(CLASSES)).to(CUSTOM_DEVICE)
        state_dict = torch.load(CUSTOM_CHECKPOINT, map_location=CUSTOM_DEVICE)
        model.load_state_dict(state_dict, strict=False)
        device = CUSTOM_DEVICE
        image_size = CUSTOM_IMAGE_SIZE
    else:
        model, _ = PretrainedBuildModel("efficientnet_v2_s", num_classes=len(CLASSES), pretrained=False)
        state_dict = torch.load(PRETRAINED_CHECKPOINT, map_location=PRETRAINED_DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(PRETRAINED_DEVICE)
        device = PRETRAINED_DEVICE
        image_size = PRETRAINED_IMAGE_SIZE

    model.eval()
    _loaded_models[key] = (model, device, image_size)
    return _loaded_models[key]


def predict(image_path, model_key):
    model, device, image_size = get_model(model_key)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)[0].cpu()

    return {name: probs[i].item() for i, name in enumerate(CLASSES)}


# ---- Visual design ----------------------------------------------------------

BG = "#F4F6FB"
CARD_BG = "#FFFFFF"
HEADER_BG = "#232946"
HEADER_FG = "#FFFFFF"
HEADER_SUB_FG = "#B8BFE0"
ACCENT = "#3B6FE0"
ACCENT_DARK = "#2B54B0"
TEXT_MAIN = "#1E2233"
TEXT_MUTED = "#6B7280"
BORDER = "#E3E7F1"
GOOD = "#1E9E63"
BAD = "#DA4453"
CLASS_COLORS = {"Fresh": "#2FA86A", "Rotten": "#C0392B", "Formalin-mixed": "#D68910"}
FONT_FAMILY = "Segoe UI"

# ---- Severity grading (rule-based, NOT learned) -----------------------------
# The models are trained to classify condition (Fresh/Rotten/Formalin-mixed) only --
# there is no severity label in the data, and the three classes are not naturally
# ordinal (Formalin-mixed is a contamination/food-safety label, not necessarily a
# "worse" visual decay state than Rotten). This maps the classifier's output onto a
# simple grading scheme as a business-rule layer on top, not a trained prediction.
# Ranking used: Fresh (best) < Rotten (spoilage, reject) < Formalin-mixed (worst --
# deliberate contamination is a food-safety hazard, not just spoiled produce).
SEVERITY_INFO = {
    "Fresh": {"grade": "A", "label": "Pass -- No Defect", "color": "#1E9E63"},
    "Rotten": {"grade": "B", "label": "Defect -- Spoilage (Reject)", "color": "#D68910"},
    "Formalin-mixed": {"grade": "C", "label": "Defect -- Contamination (Reject Immediately)", "color": "#9B1C31"},
}


def grade_result(predicted_label, confidence):
    info = SEVERITY_INFO[predicted_label]
    if confidence >= 0.80:
        certainty = "High confidence"
    elif confidence >= 0.50:
        certainty = "Moderate confidence -- spot-check recommended"
    else:
        certainty = "Low confidence -- manual inspection required"
    return {**info, "certainty": certainty}


class Card(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, **kwargs)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fruit Condition Classifier")
        self.geometry("520x760")
        self.minsize(480, 700)
        self.configure(bg=BG)
        self.image_path = None
        self.photo = None

        self._build_style()
        self._build_header()
        self._build_body()

    # -- construction -----------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TCombobox", fieldbackground="white", background="white",
                         foreground=TEXT_MAIN, arrowcolor=ACCENT, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", "white")])

        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                         font=(FONT_FAMILY, 11, "bold"), padding=(14, 10), borderwidth=0)
        style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK)])

        style.configure("Secondary.TButton", background="white", foreground=ACCENT,
                         font=(FONT_FAMILY, 10, "bold"), padding=(12, 8),
                         borderwidth=1, relief="solid")
        style.map("Secondary.TButton", background=[("active", "#EEF2FE")])

        style.configure("Confidence.Horizontal.TProgressbar", troughcolor=BORDER,
                         background=ACCENT, thickness=14, borderwidth=0)

    def _build_header(self):
        header = tk.Frame(self, bg=HEADER_BG)
        header.pack(fill="x")
        tk.Label(header, text="Fruit Condition Classifier", bg=HEADER_BG, fg=HEADER_FG,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(header, text="Pick a model, choose an image, see the prediction",
                 bg=HEADER_BG, fg=HEADER_SUB_FG, font=(FONT_FAMILY, 10)).pack(anchor="w", padx=24, pady=(0, 18))

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=18)

        # -- Model + browse card --
        controls = Card(body)
        controls.pack(fill="x", pady=(0, 14))
        inner = tk.Frame(controls, bg=CARD_BG)
        inner.pack(fill="x", padx=16, pady=14)

        tk.Label(inner, text="MODEL", bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")
        self.model_choice = tk.StringVar(value=list(MODEL_OPTIONS.keys())[0])
        ttk.Combobox(inner, textvariable=self.model_choice, values=list(MODEL_OPTIONS.keys()),
                     state="readonly", font=(FONT_FAMILY, 10)).pack(fill="x", pady=(4, 12))

        ttk.Button(inner, text="Browse Image...", style="Secondary.TButton",
                   command=self.browse).pack(fill="x")

        # -- Image preview card --
        preview_card = Card(body)
        preview_card.pack(fill="x", pady=(0, 14))
        self.preview_frame = tk.Frame(preview_card, bg=CARD_BG, height=260)
        self.preview_frame.pack(fill="x", padx=16, pady=16)
        self.preview_frame.pack_propagate(False)
        self.image_label = tk.Label(self.preview_frame, bg="#FAFBFF", fg=TEXT_MUTED,
                                     text="No image selected", font=(FONT_FAMILY, 10))
        self.image_label.pack(fill="both", expand=True)

        # -- Predict button --
        ttk.Button(body, text="Predict", style="Accent.TButton",
                   command=self.predict).pack(fill="x", pady=(0, 14))

        # -- Results card --
        self.results_card = Card(body)
        self.results_card.pack(fill="both", expand=True)
        self.results_inner = tk.Frame(self.results_card, bg=CARD_BG)
        self.results_inner.pack(fill="both", expand=True, padx=18, pady=16)
        self._render_placeholder()

    # -- rendering ----------------------------------------------------------

    def _clear_results(self):
        for widget in self.results_inner.winfo_children():
            widget.destroy()

    def _render_placeholder(self):
        self._clear_results()
        tk.Label(self.results_inner, text="Predictions will appear here",
                 bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_FAMILY, 10)).pack(pady=20)

    def browse(self):
        initial_dir = str(DATASET_TEST_DIR) if DATASET_TEST_DIR.exists() else str(PROJECT_ROOT)
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png")],
        )
        if not path:
            return
        self.image_path = path

        image = Image.open(path).convert("RGB")
        image.thumbnail((320, 228))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="", bg=CARD_BG)

    def predict(self):
        if self.image_path is None:
            self._clear_results()
            tk.Label(self.results_inner, text="Pick an image first.",
                     bg=CARD_BG, fg=BAD, font=(FONT_FAMILY, 10, "bold")).pack(pady=20)
            return

        model_key = MODEL_OPTIONS[self.model_choice.get()]
        probs = predict(self.image_path, model_key)
        predicted_label = max(probs, key=probs.get)

        parent_folder = Path(self.image_path).parent.name
        true_label = parent_folder if parent_folder in CLASSES else None

        self._render_results(predicted_label, probs, true_label)

    def _render_results(self, predicted_label, probs, true_label):
        self._clear_results()

        # True vs predicted summary row.
        summary = tk.Frame(self.results_inner, bg=CARD_BG)
        summary.pack(fill="x", pady=(0, 14))

        if true_label is None:
            badge_text, badge_color = "UNLABELED IMAGE", TEXT_MUTED
        elif true_label == predicted_label:
            badge_text, badge_color = "CORRECT", GOOD
        else:
            badge_text, badge_color = "INCORRECT", BAD

        tk.Label(summary, text=badge_text, bg=badge_color, fg="white",
                 font=(FONT_FAMILY, 9, "bold"), padx=10, pady=4).pack(anchor="w")

        true_text = true_label if true_label else "Unknown (not from labeled test set)"
        tk.Label(self.results_inner, text=f"True label:  {true_text}",
                 bg=CARD_BG, fg=TEXT_MAIN, font=(FONT_FAMILY, 10)).pack(anchor="w", pady=(6, 0))

        predicted_color = CLASS_COLORS.get(predicted_label, ACCENT)
        tk.Label(self.results_inner,
                 text=f"Predicted:  {predicted_label}  ({probs[predicted_label]:.1%} confidence)",
                 bg=CARD_BG, fg=predicted_color, font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", pady=(2, 16))

        # -- Severity grade (rule-based layer on top of the classifier output) --
        grade = grade_result(predicted_label, probs[predicted_label])
        grade_box = tk.Frame(self.results_inner, bg=grade["color"])
        grade_box.pack(fill="x", pady=(0, 16))
        grade_inner = tk.Frame(grade_box, bg=grade["color"])
        grade_inner.pack(fill="x", padx=14, pady=10)
        tk.Label(grade_inner, text=f"GRADE {grade['grade']}", bg=grade["color"], fg="white",
                 font=(FONT_FAMILY, 14, "bold")).pack(anchor="w")
        tk.Label(grade_inner, text=grade["label"], bg=grade["color"], fg="white",
                 font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(grade_inner, text=grade["certainty"], bg=grade["color"], fg="white",
                 font=(FONT_FAMILY, 9)).pack(anchor="w", pady=(4, 0))
        tk.Label(self.results_inner,
                 text="Severity grade is a rule-based mapping of the model's output, not a trained prediction.",
                 bg=CARD_BG, fg=TEXT_MUTED, font=(FONT_FAMILY, 8), wraplength=440, justify="left").pack(anchor="w", pady=(0, 14))

        tk.Label(self.results_inner, text="CLASS PROBABILITIES", bg=CARD_BG, fg=TEXT_MUTED,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w", pady=(0, 8))

        for name in CLASSES:
            row = tk.Frame(self.results_inner, bg=CARD_BG)
            row.pack(fill="x", pady=4)

            label_font = (FONT_FAMILY, 10, "bold") if name == predicted_label else (FONT_FAMILY, 10)
            tk.Label(row, text=name, bg=CARD_BG, fg=TEXT_MAIN, font=label_font, width=15, anchor="w").pack(side="left")

            bar = ttk.Progressbar(row, style="Confidence.Horizontal.TProgressbar",
                                   maximum=1.0, value=probs[name], length=180)
            bar.pack(side="left", padx=8)

            tk.Label(row, text=f"{probs[name]:.1%}", bg=CARD_BG, fg=TEXT_MAIN,
                     font=label_font, width=6, anchor="e").pack(side="left")


if __name__ == "__main__":
    App().mainloop()
