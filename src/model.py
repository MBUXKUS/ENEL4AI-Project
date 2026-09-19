from torch import nn
from torchvision import models

_BUILDERS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def build_model(name, num_classes, pretrained=True):
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_BUILDERS)}")

    fn, weights = _BUILDERS[name]
    model = fn(weights=weights if pretrained else None)

    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    elif name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    elif name == "convnext_tiny":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]

    return model, backbone_params
