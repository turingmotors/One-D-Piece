import torch.nn as nn

from modeling.modules.base_model import BaseModel

class LinearProbingModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        feature_dim = config.model.classifier.feature_dim
        num_classes = config.model.classifier.num_classes
        self.bn = nn.BatchNorm1d(feature_dim, affine=False)
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.bn(x)
        return self.linear(x)
