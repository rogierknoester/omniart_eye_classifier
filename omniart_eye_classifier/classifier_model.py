from torch import nn
from torchvision import models


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        model_ft = models.resnet18()

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_ftrs, 256),
            nn.Dropout(0.6),
            nn.Linear(256, 10)
        )

        self.layers = model_ft

    def forward(self, x):
        return self.layers(x)


