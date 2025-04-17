import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import BertTokenizer
from models.vqa_model import VQAModel
from data.dataset import ECGVQADataset
from utils.helpers import load_config

config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ECGVQADataset(config['train_json'], config['image_dir'], tokenizer, config['max_seq_length'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

model = VQAModel(config).to(device)
optimizer = Adam(model.parameters(), lr=config['learning_rate'])
criterion = CrossEntropyLoss()

model.train()
for epoch in range(config['num_epochs']):
    total_loss = 0
    for image, input_ids, attention_mask, label in train_loader:
        image, input_ids, attention_mask, label = image.to(device), input_ids.to(device), attention_mask.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image, input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), config['model_save_path'])
print("Model saved.")
