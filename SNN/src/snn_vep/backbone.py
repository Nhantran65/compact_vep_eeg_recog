from __future__ import annotations
import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    """ShallowConvNet matching the notebook architecture."""

    def __init__(
        self,
        nb_classes: int = 4,
        Chans: int = 14,
        Samples: int = 128,
        dropoutRate: float = 0.5,
    ):
        super().__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        self.conv1 = nn.Conv2d(1, 40, (1, 25), bias=True)
        self.conv2 = nn.Conv2d(40, 40, (Chans, 1), bias=True)
        self.bn = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropoutRate)
        self.feature_size = self._calculate_feature_size()
        self.fc = nn.Linear(self.feature_size, nb_classes)

    def _features_before_dropout(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        return x

    def _calculate_feature_size(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.Chans, self.Samples)
            features = self._features_before_dropout(dummy)
            feature_size = features.numel()
        self.train(was_training)
        return feature_size

    def extract_features(self, x: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        x = self._features_before_dropout(x)
        if apply_dropout:
            x = self.dropout(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x, apply_dropout=True)
        return self.fc(x)
