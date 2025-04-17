import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.vqa_model import VQAModel
from data.dataset import ECGVQADataset
from utils.helpers import load_config

config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
val_dataset = ECGVQADataset(config['val_json'], config['image_dir'], tokenizer, config['max_seq_length'])
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

model = VQAModel(config).to(device)
model.load_state_dict(torch.load(config['model_save_path']))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for image, input_ids, attention_mask, label in val_loader:
        image, input_ids, attention_mask, label = image.to(device), input_ids.to(device), attention_mask.to(device), label.to(device)
        output = model(image, input_ids, attention_mask)
        pred = torch.argmax(output, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Validation Accuracy: {100 * correct / total:.2f}%")