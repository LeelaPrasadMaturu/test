# test.py
import torch
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from utils.helpers import load_config
from models.vqa_model import ClinicalVQAModel
from data.dataset import ClinicalECGDataset
from transformers import BertTokenizer

config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

# Load model
model = ClinicalVQAModel(config).to(device)
model.load_state_dict(torch.load(config['model_save_path']))
model.eval()

# Load test data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_dataset = ClinicalECGDataset(
    config['test_json'],
    config['image_dir'],
    tokenizer,
    config['max_seq_length']
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    num_workers=min(4, os.cpu_count())
)

# Testing loop
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        
        outputs = model(images, input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

# Generate clinical report
print("\nClinical Test Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["Negative", "Positive"],
    digits=4
))