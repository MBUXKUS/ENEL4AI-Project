# Evaluation Report (efficientnet_v2_s, condition)

Checkpoint: `C:\Users\kevol\Desktop\AI_Project\AI_Project\outputs\checkpoints\efficientnet_v2_s_condition_best.pt`
Test-time augmentation: with TTA

**Test accuracy: 83.0%**

Macro precision: 84.4% | Macro recall: 83.1% | Macro F1: 83.4%

## Per-class metrics

| Class | Recall | Precision | F1 | Support |
|---|---|---|---|---|
| Formalin-mixed | 88.6% | 76.1% | 81.9% | 474 |
| Fresh | 70.4% | 82.5% | 75.9% | 415 |
| Rotten | 90.3% | 94.8% | 92.5% | 360 |

## Per-class precision / recall / F1

![Per-class metrics](figures/per_class_metrics_efficientnet_v2_s_condition.png)

## Confusion matrix

![Confusion matrix](figures/confusion_matrix_efficientnet_v2_s_condition.png)

## ROC curves (one-vs-rest)

![ROC curves](figures/roc_curve_efficientnet_v2_s_condition.png)
