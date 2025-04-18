# models/fusion.py
import torch
import torch.nn as nn

class ClinicalCrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(emb_dim, emb_dim)
        self.text_proj = nn.Linear(emb_dim, emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, num_heads)
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, img_feat, text_feat):
        img_proj = self.img_proj(img_feat).unsqueeze(1)
        text_proj = self.text_proj(text_feat).unsqueeze(1)
        
        attended, _ = self.attention(
            query=img_proj,
            key=text_proj,
            value=text_proj
        )
        return self.norm(attended.squeeze(1) + img_feat)

class ClinicalFusion(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_heads=8):
        super().__init__()
        
        self.cross_attn = ClinicalCrossAttention(emb_dim, num_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 2)
        )
        
    def forward(self, img_feat, text_feat):
        # Cross-modal attention
        attended_img = self.cross_attn(img_feat, text_feat)
        
        # Concatenate and classify
        combined = torch.cat([attended_img, text_feat], dim=1)
        return self.classifier(combined)