import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention. Cheap, helps the network
    weight *which* channels carry the diagnostic signal (e.g. a subtle
    color/sheen shift for formalin) rather than treating all channels equally."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class ResConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ shortcut) -> SE -> ReLU -> MaxPool2x2

    Residual connection eases optimization; 1x1 projection shortcut is used
    whenever channel count changes between input and output.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return self.pool(out)


class VGGLiteGAP(nn.Module):
    """5 residual+SE conv blocks (3->32->64->128->256->512) + GAP&GMP concat
    + dropout MLP head.

    Drop-in replacement for the original VGGLiteGAP: same constructor
    signature plus one new optional argument.

    aux_classes: if set (e.g. to the number of fruit types), adds a second
    head that predicts fruit identity from the shared backbone features.
    This is a multi-task learning trick: fruit identity and condition labels
    come from the same images, so the fruit label is "free" supervision that
    can regularize the shared backbone and often improves the main head's
    accuracy on small datasets. When set,
    forward() returns (logits, aux_logits) and the training loop should do:
        loss = criterion(logits, condition_labels) \
             + aux_weight * criterion(aux_logits, fruit_labels)
    with aux_weight around 0.2-0.5. Only `logits` matters at inference.
    Leave aux_classes=None to use exactly like the original single-head model.
    """

    def __init__(self, num_classes=3, dropout=0.5, aux_classes=None):
        super().__init__()
        channels = [3, 32, 64, 128, 256, 512]
        self.blocks = nn.Sequential(*[ResConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        feat_dim = channels[-1] * 2  # GAP (average texture) + GMP (peak defect signal) concatenated

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.aux_classifier = None
        if aux_classes is not None:
            self.aux_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, aux_classes),
            )

        self._init_weights()

    def forward(self, x):
        x = self.blocks(x)
        feats = torch.cat([self.gap(x).flatten(1), self.gmp(x).flatten(1)], dim=1)
        logits = self.classifier(feats)
        if self.aux_classifier is not None:
            aux_logits = self.aux_classifier(feats)
            return logits, aux_logits
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
