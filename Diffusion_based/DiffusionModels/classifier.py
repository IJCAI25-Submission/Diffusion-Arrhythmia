import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(classifier, self).__init__()
        self.signal_layer = nn.Sequential(
            nn.Conv1d(12, 3, 6, 1, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.lorenz_layer = nn.Sequential(
            nn.Conv2d(12, 3, 6, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.frequency_layer = nn.Sequential(
            nn.Conv2d(12, 3, 6, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(3618, 1024),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(256, num_classes)
    def forward(self, signal, lorenz, frequency):
        signal = self.signal_layer(signal)
        lorenz = self.lorenz_layer(lorenz)
        frequency = self.frequency_layer(frequency)

        com = torch.cat([signal, lorenz, frequency], dim=1)

        x = self.fc1(com)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CrossAttnClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CrossAttnClassifier, self).__init__()
        self.signal_layer = nn.Sequential(
            nn.Conv1d(96, 3, 6, 1, 1),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(933, 512),
        )
        self.lorenz_layer = nn.Sequential(
            nn.Conv2d(192, 64, 6, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, 512),
        )
        self.frequency_layer = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(576, 512),
        )

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(512)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, signal, lorenz, frequency):
        signal = self.signal_layer(signal)
        lorenz = self.lorenz_layer(lorenz)
        frequency = self.frequency_layer(frequency)

        modalities = torch.stack([signal, lorenz, frequency], dim=1)

        attn_output, _ = self.attention(modalities, modalities, modalities)
        attn_output = self.attention_norm(attn_output)
        fused_features = torch.mean(attn_output, dim=1)

        x = self.fc1(fused_features)
        x = self.fc2(x)

        return x
