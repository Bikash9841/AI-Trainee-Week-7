import torch
import torch.nn as nn


class Alexnet(nn.Module):

    def __init__(self, in_channel=3, out_classes=10):
        super().__init__()

        self.in_channel = in_channel
        self.out_classes = out_classes

        self.model = nn.Sequential(

            # conv->maxpool->conv->maxpool->3*conv->maxpool->3*fc->outputlayer
            nn.Conv2d(self.in_channel, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.001, 0.75, 2),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, padding='same'),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.001, 0.75, 2),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            # fully connected layer
            nn.Flatten(start_dim=1),
            nn.Linear(256*6*6, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, self.out_classes),

        )

    def forward(self, x):
        return self.model(x)
