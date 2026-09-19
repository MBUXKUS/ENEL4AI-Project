# AI_Project

## Usage

Install dependencies:

```
pip install torch torchvision matplotlib numpy pillow
```

Train a model:

```
python src/train.py            # EfficientNetV2-S (transfer learning)
python src_custom/train.py     # VGGLiteGAP (from scratch)
```

Evaluate a trained model:

```
python src/evaluate.py
python src_custom/evaluate.py
```

Run a single prediction from the command line:

```
python src_custom/predict.py --image path/to/image.jpg
```

Launch the desktop GUI:

```
python scripts/predict_gui.py
```
