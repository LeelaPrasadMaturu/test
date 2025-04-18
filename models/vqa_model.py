# models/vqa_model.py
import torch.nn as nn
from models.image_encoder import ClinicalImageEncoder
from models.text_encoder import ClinicalTextEncoder
from models.fusion import ClinicalFusion

class ClinicalVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.image_encoder = ClinicalImageEncoder(config['embedding_dim'])
        self.text_encoder = ClinicalTextEncoder(
            config['embedding_dim'],
            freeze_bert=config.get('freeze_bert', True)
        )
        
        self.fusion = ClinicalFusion(
            emb_dim=config['embedding_dim'],
            hidden_dim=config['fusion_hidden_dim'],
            num_heads=config.get('num_attention_heads', 8)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        text_feat = self.text_encoder(input_ids, attention_mask)
        return self.fusion(img_feat, text_feat)