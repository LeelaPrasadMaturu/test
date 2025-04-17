import json
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

class ECGVQADataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, max_seq_len, transform=None):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample['question']
        answer = 1 if sample['answer'][0] == 'yes' else 0
        image_path = os.path.join(self.image_dir, sample['ecg_path'][0] + ".png")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        encoding = self.tokenizer(question, padding='max_length', max_length=self.max_seq_len,
                                  truncation=True, return_tensors='pt')
        return image, encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(answer)

