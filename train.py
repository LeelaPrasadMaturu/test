# train.py
import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from utils.helpers import load_config
from models.vqa_model import ClinicalVQAModel
from data.dataset import ClinicalECGDataset
from transformers import BertTokenizer

# Configuration
config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Initialize components
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = ClinicalECGDataset(
    config['train_json'],
    config['image_dir'],
    tokenizer,
    config['max_seq_length'],
    is_train=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True
)

model = ClinicalVQAModel(config).to(device)
optimizer = AdamW(
    model.parameters(),
    lr=float(config['learning_rate']),
    weight_decay=config['weight_decay'],
    fused=True
)
criterion = CrossEntropyLoss()
scaler = GradScaler()

# Checkpoint handling
def load_checkpoint():
    if os.path.exists(config['checkpoint_path']):
        checkpoint = torch.load(config['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        return checkpoint['epoch'] + 1
    return 0

start_epoch = load_checkpoint()

# Training loop
model.train()
for epoch in range(start_epoch, config['num_epochs']):
    epoch_loss = 0
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels) / config['grad_accum_steps']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % config['grad_accum_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Logging
        epoch_loss += loss.item() * config['grad_accum_steps']
        if step % 10 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

    # Epoch statistics
    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} Completed | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'loss': avg_loss,
    }, config['checkpoint_path'])

# Final save
torch.save(model.state_dict(), config['model_save_path'])
print("Training completed.")