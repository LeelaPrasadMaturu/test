# data/dataset.py
import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

class ClinicalECGDataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, max_seq_len, is_train=False):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
            
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        
        # Medical-grade transformations
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02)
                )
            ], p=0.5 if is_train else 0),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1
                )
            ], p=0.3 if is_train else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(
            self.image_dir,
            f"{sample['ecg_path'][0]}.png"
        )
        
        # Load and transform image
        image = Image.open(image_path)
        image = self.transform(image)
        
        # Process question
        question = sample['question']
        encoding = self.tokenizer(
            question,
            padding='max_length',
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors='pt'
        )
        
        # Process answer
        answer = 1 if sample['answer'][0].lower() == 'yes' else 0
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(answer, dtype=torch.long)
        }