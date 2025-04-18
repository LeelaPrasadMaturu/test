# models/image_encoder.py
import torch
import torch.nn as nn

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.add_coords = True
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        
    def forward(self, x):
        if self.add_coords:
            batch_size, _, h, w = x.size()
            xx_channel = torch.linspace(-1, 1, w, device=x.device).repeat(h, 1)
            yy_channel = torch.linspace(-1, 1, h, device=x.device).repeat(w, 1).t()
            xx_channel = xx_channel.unsqueeze(0).unsqueeze(0)
            yy_channel = yy_channel.unsqueeze(0).unsqueeze(0)
            xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
            x = torch.cat([x, xx_channel, yy_channel], dim=1)
            
        return self.conv(x)

class ClinicalImageEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.base = nn.Sequential(
            CoordConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        features = self.base(x).squeeze(-1).squeeze(-1)
        return self.fc(features)