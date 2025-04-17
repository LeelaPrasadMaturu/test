import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion import FusionModule

class VQAModel(nn.Module):
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.image_encoder = ImageEncoder(config['embedding_dim'])
        self.text_encoder = TextEncoder(config['embedding_dim'])
        self.fusion = FusionModule(
            input_dim=2*config['embedding_dim'],
            hidden_dim=config['fusion_hidden_dim'],
            output_dim=config['num_classes']
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        text_feat = self.text_encoder(input_ids, attention_mask)
        return self.fusion(img_feat, text_feat)
