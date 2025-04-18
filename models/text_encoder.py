# models/text_encoder.py
import torch.nn as nn
from transformers import BertModel, BertConfig

class ClinicalTextEncoder(nn.Module):
    def __init__(self, embedding_dim, freeze_bert=True):
        super().__init__()
        
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(config)
        self.freeze = freeze_bert
        
        if self.freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)