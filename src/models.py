import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        return (weights * x).sum(dim=1)    # [B, D]

class CRNNWithAttn(nn.Module):
    def __init__(self,  pretrained=True, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()
        # 1. Pretrained ResNet18
        if pretrained:
          resnet = models.resnet18(weights='DEFAULT')
        else:
          resnet = models.resnet18()
        # Adapt first conv to accept 1-channel input
        w = resnet.conv1.weight.data.clone()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data[:, 0] = w[:, 0]
        # Remove final pooling & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Bi-GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=512,          # ResNet last block outputs 512 channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )

        # 3. Attention pooling
        self.attn_pool = AttentionPool(hidden_size*2)

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        feat = self.backbone(x)            # [B, 512, F', T']
        feat = feat.mean(dim=2)            # collapse freq → [B,512,T']
        feat = feat.permute(0,2,1)         # → [B,T',512]

        out, _ = self.gru(feat)            # → [B,T',2*hidden_size]
        pooled = self.attn_pool(out)       # → [B,2*hidden_size]
        return self.classifier(pooled)     # → [B,1]

