import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, emb_dim=128, num_classes=7, p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*3, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, num_classes)
        )

    def forward(self, zv, za, zt):
        fused = torch.cat([zv, za, zt], dim=1)
        return self.mlp(fused)
