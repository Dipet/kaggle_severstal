import torch.nn as nn

from catalyst.contrib import registry

import segmentation_models_pytorch as smp


@registry.Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.FPN('dpn131', encoder_weights='imagenet', classes=4, activation=None)

    def forward(self, x):
        return self.model(x)
