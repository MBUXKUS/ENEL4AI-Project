# Evaluation Report (condition)

Checkpoint: `C:\Users\kevol\Desktop\AI_Project\AI_Project\outputs\checkpoints_custom\custom_condition_best.pt`

**Test accuracy: 80.5%**

Macro precision: 82.1% | Macro recall: 81.1% | Macro F1: 81.4%

## Per-class metrics

| Class | Recall | Precision | F1 | Support |
|---|---|---|---|---|
| Formalin-mixed | 73.8% | 77.8% | 75.8% | 474 |
| Fresh | 82.2% | 72.1% | 76.8% | 415 |
| Rotten | 87.2% | 96.3% | 91.5% | 360 |

## Per-class precision / recall / F1

![Per-class metrics](figures/per_class_metrics_condition.png)

## Confusion matrix

![Confusion matrix](figures/confusion_matrix_condition.png)

## ROC curves (one-vs-rest)

![ROC curves](figures/roc_curve_condition.png)
