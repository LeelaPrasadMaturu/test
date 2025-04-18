# validate.py
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
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

# Load data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
val_dataset = ClinicalECGDataset(
    config['val_json'],
    config['image_dir'],
    tokenizer,
    config['max_seq_length']
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    num_workers=min(4, os.cpu_count())
)

# Validation metrics
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images, input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        
        all_probs.extend(probs[:,1].cpu().numpy())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"Clinical Validation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")